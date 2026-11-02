// Standalone serial regression / construction-count benchmark for MFEM #5535.
//
// From a directory holding the MFEM source in mfem/ and a serial CMake build in build/:
// c++ -std=c++17 -O0 -g0 -Ibuild -Imfem selective_test.cpp build/libmfem.a -o selective_test
// ./selective_test
// Negative tests, each in its own subprocess with core dumps disabled:
// ./selective_test --invalid-mask 1 1
// ./selective_test --invalid-mask 0 2
// ./selective_test --invalid-mask 2 0
// ./selective_test --invalid-mask 3 3
// Expect MFEM's validation failure / abort, not our "was accepted" diagnostic.
//
// The polynomial mass oracle uses exact Q1/Q2 tensor-product integral tables,
// not this integrator's quadrature or a full gradient as its reference.
// These are numerical regression tests, not formal verification or speed claims.

#include "mfem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace mfem;

namespace
{

/// Accumulate failed assertions without relying on assert()/NDEBUG.
struct Checks
{
   int assertions = 0;
   int failures = 0;
   int cases = 0;
   int masks = 0;
   int differences = 0;

   void Require(bool condition, const std::string &message)
   {
      ++assertions;
      if (!condition)
      {
         ++failures;
         std::cerr << "FAIL: " << message << '\n';
      }
   }

   void Near(const DenseMatrix &actual, const DenseMatrix &expected,
             const std::string &message, real_t tolerance = 2e-11)
   {
      if (actual.Height() != expected.Height() ||
          actual.Width() != expected.Width())
      {
         Require(false, message + " (matrix dimensions)");
         return;
      }
      real_t error = 0.0, scale = 0.0;
      bool finite = true;
      for (int i = 0; i < actual.Height(); ++i)
      {
         for (int j = 0; j < actual.Width(); ++j)
         {
            finite = finite && std::isfinite(actual(i,j)) &&
                     std::isfinite(expected(i,j));
            error = std::max(error, std::abs(actual(i,j) - expected(i,j)));
            scale = std::max(scale, std::abs(expected(i,j)));
         }
      }
      Require(finite && error <= tolerance * (1.0 + scale),
              message + " (max error=" + std::to_string(error) + ")");
   }
};

using Blocks = std::array<DenseMatrix, 4>;
using EssentialDofs = std::array<Array<int>, 2>;

/// Encode a row-major subset; signed nonzero entries also mean selected.
Array2D<int> Mask(int bits, bool signed_entries = false)
{
   Array2D<int> mask(2, 2);
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         mask(i,j) = (bits & (1 << (2*i+j))) ?
                     (signed_entries ? ((i+j)%2 ? -7 : 3) : 1) : 0;
      }
   }
   return mask;
}

int BitCount(int bits)
{
   int count = 0;
   for (int i = 0; i < 4; ++i) { count += (bits >> i) & 1; }
   return count;
}

/// Copy an operator by applying it to every basis vector.
DenseMatrix Materialize(const Operator &op, bool transpose = false)
{
   const int height = transpose ? op.Width() : op.Height();
   const int width = transpose ? op.Height() : op.Width();
   DenseMatrix matrix(height, width);
   Vector basis(width), column(height);
   for (int j = 0; j < width; ++j)
   {
      basis = 0.0;
      basis(j) = 1.0;
      column = std::numeric_limits<real_t>::quiet_NaN();
      if (transpose) { op.MultTranspose(basis, column); }
      else { op.Mult(basis, column); }
      for (int i = 0; i < height; ++i) { matrix(i,j) = column(i); }
   }
   return matrix;
}

DenseMatrix Transposed(const DenseMatrix &matrix)
{
   DenseMatrix transpose(matrix.Width(), matrix.Height());
   for (int i = 0; i < matrix.Height(); ++i)
   {
      for (int j = 0; j < matrix.Width(); ++j)
      { transpose(j,i) = matrix(i,j); }
   }
   return transpose;
}

/// A full gradient's blocks must be copied before the next GetGradient call.
Blocks Snapshot(const BlockOperator &op, Checks &checks,
                const std::string &where)
{
   Blocks copy;
   for (int i = 0; i < 2; ++i)
   {
      for (int j = 0; j < 2; ++j)
      {
         checks.Require(!op.IsZeroBlock(i,j), where + " full block present");
         if (!op.IsZeroBlock(i,j))
         {
            copy[2*i+j] = Materialize(op.GetBlock(i,j));
            copy[2*i+j] *= op.GetBlockCoef(i,j);
         }
      }
   }
   return copy;
}

/// Track completed local matrices separately from quadrature or wall time.
class TrackedIntegrator : public BlockNonlinearFormIntegrator
{
public:
   const bool honor_hint;
   std::size_t calls = 0;
   std::size_t constructions = 0;
   std::size_t stale_off_mask = 0;
   std::size_t hint_updates = 0;

   explicit TrackedIntegrator(bool honor) : honor_hint(honor) { }

   void SetGradientBlockMask(const Array2D<int> &mask) override
   {
      ++hint_updates;
      BlockNonlinearFormIntegrator::SetGradientBlockMask(mask);
   }

   bool Enabled(int row, int col) const
   { return IsGradientBlockEnabled(row, col); }

   bool HintIsEmpty() const
   {
      return gradient_block_mask.NumRows() == 0 &&
             gradient_block_mask.NumCols() == 0;
   }

protected:
   /// Leave omitted DenseMatrix objects untouched, including stale contents.
   void Begin(const int sizes[2], const Array2D<DenseMatrix *> &matrices)
   {
      ++calls;
      for (int i = 0; i < 2; ++i)
      {
         for (int j = 0; j < 2; ++j)
         {
            if (honor_hint && !IsGradientBlockEnabled(i,j))
            {
               if (matrices(i,j)->Height() != 0) { ++stale_off_mask; }
               continue;
            }
            matrices(i,j)->SetSize(sizes[i], sizes[j]);
            *matrices(i,j) = 0.0;
            ++constructions;
         }
      }
   }

   void Add(const Vector shape[2], const real_t coefficients[2][2],
            real_t weight, const Array2D<DenseMatrix *> &matrices)
   {
      for (int i = 0; i < 2; ++i)
      {
         for (int j = 0; j < 2; ++j)
         {
            if (honor_hint && !IsGradientBlockEnabled(i,j)) { continue; }
            DenseMatrix &matrix = *matrices(i,j);
            for (int r = 0; r < shape[i].Size(); ++r)
            {
               for (int c = 0; c < shape[j].Size(); ++c)
               {
                  matrix(r,c) += weight * coefficients[i][j] *
                                 shape[i](r) * shape[j](c);
               }
            }
         }
      }
   }
};

/// Weak residual (u^2+3uv, 5uv+v^2), for domain or boundary elements.
class PolynomialMassIntegrator : public TrackedIntegrator
{
public:
   explicit PolynomialMassIntegrator(bool honor = true)
      : TrackedIntegrator(honor) { }

   void AssembleElementVector(
      const Array<const FiniteElement *> &fe, ElementTransformation &tr,
      const Array<const Vector *> &state, const Array<Vector *> &residual)
   override
   {
      Vector shape[2] = { Vector(fe[0]->GetDof()), Vector(fe[1]->GetDof()) };
      for (int i = 0; i < 2; ++i)
      {
         residual[i]->SetSize(fe[i]->GetDof());
         *residual[i] = 0.0;
      }
      const IntegrationRule &rule = IntRules.Get(
         fe[0]->GetGeomType(),
         3 * std::max(fe[0]->GetOrder(), fe[1]->GetOrder()) + tr.OrderW());
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = rule.IntPoint(q);
         tr.SetIntPoint(&ip);
         for (int i = 0; i < 2; ++i) { fe[i]->CalcShape(ip, shape[i]); }
         const real_t u = shape[0] * *state[0];
         const real_t v = shape[1] * *state[1];
         const real_t weight = ip.weight * tr.Weight();
         residual[0]->Add(weight * (u*u + 3*u*v), shape[0]);
         residual[1]->Add(weight * (5*u*v + v*v), shape[1]);
      }
   }

   void AssembleElementGrad(
      const Array<const FiniteElement *> &fe, ElementTransformation &tr,
      const Array<const Vector *> &state,
      const Array2D<DenseMatrix *> &matrices) override
   {
      const int sizes[2] = { fe[0]->GetDof(), fe[1]->GetDof() };
      Begin(sizes, matrices);
      Vector shape[2] = { Vector(sizes[0]), Vector(sizes[1]) };
      const IntegrationRule &rule = IntRules.Get(
         fe[0]->GetGeomType(),
         3 * std::max(fe[0]->GetOrder(), fe[1]->GetOrder()) + tr.OrderW());
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = rule.IntPoint(q);
         tr.SetIntPoint(&ip);
         for (int i = 0; i < 2; ++i) { fe[i]->CalcShape(ip, shape[i]); }
         // Both state fields are read even when only block (0,0) is selected.
         const real_t u = shape[0] * *state[0];
         const real_t v = shape[1] * *state[1];
         const real_t jacobian[2][2] =
         { { 2*u + 3*v, 3*u }, { 5*v, 5*u + 2*v } };
         Add(shape, jacobian, ip.weight * tr.Weight(), matrices);
      }
   }
};

/// Linear coupled trace mass; interior traces average the two sides.
class FaceMassIntegrator : public TrackedIntegrator
{
   /// Trace vectors match the concatenated element DOF ordering on a face.
   void Shapes(const Array<const FiniteElement *> &fe1,
               const Array<const FiniteElement *> &fe2,
               FaceElementTransformations &tr, Vector shape[2])
   {
      for (int i = 0; i < 2; ++i)
      {
         const int n1 = fe1[i]->GetDof();
         const int n2 = tr.Elem2No >= 0 ? fe2[i]->GetDof() : 0;
         shape[i].SetSize(n1+n2);
         Vector first(shape[i].GetData(), n1);
         fe1[i]->CalcShape(tr.GetElement1IntPoint(), first);
         if (n2)
         {
            Vector second(shape[i].GetData()+n1, n2);
            fe2[i]->CalcShape(tr.GetElement2IntPoint(), second);
            shape[i] *= 0.5;
         }
      }
   }

public:
   explicit FaceMassIntegrator(bool honor) : TrackedIntegrator(honor) { }

   void AssembleFaceVector(
      const Array<const FiniteElement *> &fe1,
      const Array<const FiniteElement *> &fe2, FaceElementTransformations &tr,
      const Array<const Vector *> &state, const Array<Vector *> &residual)
   override
   {
      for (int i = 0; i < 2; ++i)
      {
         residual[i]->SetSize(state[i]->Size());
         *residual[i] = 0.0;
      }
      Vector shape[2];
      const IntegrationRule &rule = IntRules.Get(tr.GetGeometryType(), 6);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = rule.IntPoint(q);
         tr.SetAllIntPoints(&ip);
         Shapes(fe1, fe2, tr, shape);
         const real_t u = shape[0] * *state[0];
         const real_t v = shape[1] * *state[1];
         const real_t weight = ip.weight * tr.Weight();
         residual[0]->Add(weight * (2*u + 3*v), shape[0]);
         residual[1]->Add(weight * (5*u + 7*v), shape[1]);
      }
   }

   void AssembleFaceGrad(
      const Array<const FiniteElement *> &fe1,
      const Array<const FiniteElement *> &fe2, FaceElementTransformations &tr,
      const Array<const Vector *> &state,
      const Array2D<DenseMatrix *> &matrices) override
   {
      const int sizes[2] = { state[0]->Size(), state[1]->Size() };
      Begin(sizes, matrices);
      const real_t coefficients[2][2] = { { 2, 3 }, { 5, 7 } };
      Vector shape[2];
      const IntegrationRule &rule = IntRules.Get(tr.GetGeometryType(), 6);
      for (int q = 0; q < rule.GetNPoints(); ++q)
      {
         const IntegrationPoint &ip = rule.IntPoint(q);
         tr.SetAllIntPoints(&ip);
         Shapes(fe1, fe2, tr, shape);
         Add(shape, coefficients, ip.weight * tr.Weight(), matrices);
      }
   }
};

/// Exact integrals of the nodal bases on [0,1], at nodes 0, 1/2, 1.
real_t Mass1D(int p, int q, int i, int j)
{
   static const real_t m11[2][2] =
   { { 1.0/3, 1.0/6 }, { 1.0/6, 1.0/3 } };
   static const real_t m12[2][3] =
   { { 1.0/6, 1.0/3, 0 }, { 0, 1.0/3, 1.0/6 } };
   static const real_t m22[3][3] =
   {
      { 2.0/15, 1.0/15, -1.0/30 },
      { 1.0/15, 8.0/15, 1.0/15 },
      { -1.0/30, 1.0/15, 2.0/15 }
   };
   if (p == 1 && q == 1) { return m11[i][j]; }
   if (p == 1 && q == 2) { return m12[i][j]; }
   if (p == 2 && q == 1) { return m12[j][i]; }
   return m22[i][j];
}

/// Materialize the FE constraint map, with null meaning the identity.
DenseMatrix Prolongation(FiniteElementSpace &space)
{
   const Operator *op = space.GetProlongationMatrix();
   if (op) { return Materialize(*op); }
   DenseMatrix identity(space.GetVSize(), space.GetTrueVSize());
   identity = 0.0;
   for (int i = 0; i < identity.Height(); ++i) { identity(i,i) = 1.0; }
   return identity;
}

/// Assemble exact unweighted mass blocks and apply P_row^T M P_col explicitly.
Blocks ExactMass(Array<FiniteElementSpace *> &spaces)
{
   Blocks mass;
   DenseMatrix prolongation[2] =
   { Prolongation(*spaces[0]), Prolongation(*spaces[1]) };
   Mesh &mesh = *spaces[0]->GetMesh();
   for (int row = 0; row < 2; ++row)
   {
      for (int col = 0; col < 2; ++col)
      {
         DenseMatrix local(spaces[row]->GetVSize(), spaces[col]->GetVSize());
         local = 0.0;
         for (int e = 0; e < mesh.GetNE(); ++e)
         {
            const FiniteElement &rfe = *spaces[row]->GetFE(e);
            const FiniteElement &cfe = *spaces[col]->GetFE(e);
            const int p = rfe.GetOrder(), q = cfe.GetOrder();
            Array<int> rdofs, cdofs;
            spaces[row]->GetElementDofs(e, rdofs);
            spaces[col]->GetElementDofs(e, cdofs);
            // All oracle meshes consist of affine axis-aligned rectangles.
            const real_t area = mesh.GetElementVolume(e);
            for (int i = 0; i < rdofs.Size(); ++i)
            {
               const IntegrationPoint &ri = rfe.GetNodes().IntPoint(i);
               const int ix = std::lround(p*ri.x), iy = std::lround(p*ri.y);
               for (int j = 0; j < cdofs.Size(); ++j)
               {
                  const IntegrationPoint &cj = cfe.GetNodes().IntPoint(j);
                  const int jx = std::lround(q*cj.x), jy = std::lround(q*cj.y);
                  const int r = rdofs[i] < 0 ? -1-rdofs[i] : rdofs[i];
                  const int c = cdofs[j] < 0 ? -1-cdofs[j] : cdofs[j];
                  const int sign = (rdofs[i] < 0 ? -1 : 1) *
                                   (cdofs[j] < 0 ? -1 : 1);
                  local(r,c) += sign * area * Mass1D(p,q,ix,jx) *
                                Mass1D(p,q,iy,jy);
               }
            }
         }
         const DenseMatrix &pr = prolongation[row], &pc = prolongation[col];
         DenseMatrix product(local.Height(), pc.Width());
         product = 0.0;
         for (int i = 0; i < local.Height(); ++i)
         {
            for (int k = 0; k < local.Width(); ++k)
            {
               for (int j = 0; j < pc.Width(); ++j)
               { product(i,j) += local(i,k) * pc(k,j); }
            }
         }
         DenseMatrix &result = mass[2*row+col];
         result.SetSize(pr.Width(), pc.Width());
         result = 0.0;
         for (int i = 0; i < result.Height(); ++i)
         {
            for (int k = 0; k < pr.Height(); ++k)
            {
               for (int j = 0; j < result.Width(); ++j)
               { result(i,j) += pr(k,i) * product(k,j); }
            }
         }
      }
   }
   return mass;
}

/// Independently scale exact mass blocks and impose the essential DOF rule.
Blocks Oracle(const Blocks &mass, real_t u, real_t v,
              const EssentialDofs &essential, int multiplicity)
{
   Blocks expected = mass;
   // Kept separate from the integrator's pointwise derivative implementation.
   const real_t coefficients[4] = { 2*u+3*v, 3*u, 5*v, 5*u+2*v };
   for (int row = 0; row < 2; ++row)
   {
      for (int col = 0; col < 2; ++col)
      {
         DenseMatrix &block = expected[2*row+col];
         block *= multiplicity * coefficients[2*row+col];
         for (int r : essential[row])
         { for (int c = 0; c < block.Width(); ++c) { block(r,c) = 0.0; } }
         for (int c : essential[col])
         { for (int r = 0; r < block.Height(); ++r) { block(r,c) = 0.0; } }
         if (row == col)
         { for (int r : essential[row]) { block(r,r) = 1.0; } }
      }
   }
   return expected;
}

/// Modes: no BCs, different boundary sides, all boundary sides, all true DOFs.
EssentialDofs SetBC(BlockNonlinearForm &form, int mode)
{
   EssentialDofs essential;
   Array<int> marker[2];
   Array<Array<int> *> markers(2), lists(2);
   Array<Vector *> rhs(2);
   rhs = nullptr;
   for (int i = 0; i < 2; ++i)
   {
      FiniteElementSpace &space = *form.FESpace(i);
      const Array<int> &attributes = space.GetMesh()->bdr_attributes;
      marker[i].SetSize(attributes.Size() ? attributes.Max() : 0);
      marker[i] = mode == 2 ? 1 : 0;
      if (mode == 1 && marker[i].Size() > i) { marker[i][i] = 1; }
      markers[i] = &marker[i];
      space.GetEssentialTrueDofs(marker[i], essential[i]);
      if (mode == 3)
      {
         essential[i].SetSize(space.GetTrueVSize());
         for (int j = 0; j < essential[i].Size(); ++j) { essential[i][j] = j; }
      }
      lists[i] = &essential[i];
   }
   if (mode == 3) { form.SetEssentialTrueDofs(lists, rhs); }
   else { form.SetEssentialBC(markers, rhs); }
   return essential;
}

void ConstantState(BlockVector &state, real_t u, real_t v)
{
   state.GetBlock(0) = u;
   state.GetBlock(1) = v;
   state.SyncFromBlocks();
}

/// Verify both individual blocks and the complete BlockOperator action.
void CheckSelection(const BlockOperator &op, int bits, const Blocks &expected,
                    Checks &checks, const std::string &where)
{
   ++checks.masks;
   checks.Require(op.NumRowBlocks() == 2 && op.NumColBlocks() == 2,
                  where + " block layout");
   DenseMatrix combined(op.Height(), op.Width());
   combined = 0.0;
   for (int row = 0; row < 2; ++row)
   {
      for (int col = 0; col < 2; ++col)
      {
         const int index = 2*row+col;
         const bool selected = bits & (1 << index);
         const std::string block = where + " block(" + std::to_string(row) +
                                   "," + std::to_string(col) + ")";
         checks.Require(bool(op.IsZeroBlock(row,col)) == !selected,
                        block + " presence / no stale block");
         if (!selected) { continue; }
         if (!op.IsZeroBlock(row,col))
         {
            DenseMatrix actual = Materialize(op.GetBlock(row,col));
            actual *= op.GetBlockCoef(row,col);
            checks.Near(actual, expected[index], block);
         }
         for (int i = 0; i < expected[index].Height(); ++i)
         {
            for (int j = 0; j < expected[index].Width(); ++j)
            {
               combined(op.RowOffsets()[row]+i, op.ColOffsets()[col]+j) =
                  expected[index](i,j);
            }
         }
      }
   }
   checks.Near(Materialize(op), combined, where + " complete operator");
   checks.Near(Materialize(op, true), Transposed(combined),
               where + " MultTranspose consistency");
}

/// Check callback hints, including clearing after assembly, and local work.
void Request(BlockNonlinearForm &form, const Vector &state, int bits,
             const Blocks &expected,
             const std::vector<TrackedIntegrator *> &integrators,
             Checks &checks, const std::string &where,
             bool empty_mask = false, bool unmasked = false)
{
   std::vector<std::size_t> calls, constructions, updates;
   for (auto *integrator : integrators)
   {
      calls.push_back(integrator->calls);
      constructions.push_back(integrator->constructions);
      updates.push_back(integrator->hint_updates);
   }
   Array2D<int> mask = empty_mask ? Array2D<int>() : Mask(bits, true);
   BlockOperator &gradient = unmasked ?
      dynamic_cast<BlockOperator &>(form.GetGradient(state)) :
      form.GetGradient(state, mask);
   CheckSelection(gradient, bits, expected, checks, where);
   for (std::size_t i = 0; i < integrators.size(); ++i)
   {
      const TrackedIntegrator &integrator = *integrators[i];
      const std::size_t visits = integrator.calls - calls[i];
      const std::size_t blocks = integrator.honor_hint ? BitCount(bits) : 4;
      checks.Require(integrator.constructions - constructions[i] ==
                     visits * blocks, where + " local construction count");
      checks.Require(integrator.HintIsEmpty(), where + " hint cleared");
      checks.Require(integrator.hint_updates >= updates[i]+2,
                     where + " hint supplied and reset");
   }
}

/// Exhaust all subsets, then explicitly exercise reuse and API transitions.
void ExerciseMasks(BlockNonlinearForm &form, const Vector &state,
                   const Blocks &expected,
                   const std::vector<TrackedIntegrator *> &integrators,
                   Checks &checks, const std::string &where)
{
   Request(form, state, 15, expected, integrators, checks, where+" full",
           false, true);
   for (int bits = 0; bits < 16; ++bits)
   {
      Request(form, state, bits, expected, integrators, checks,
              where + " mask=" + std::to_string(bits));
   }
   // Includes mask reuse, diagonal/off-diagonal changes, and an empty result.
   for (int bits : {1, 1, 2, 4, 8, 0})
   {
      Request(form, state, bits, expected, integrators, checks,
              where + " transition=" + std::to_string(bits));
   }
   Request(form, state, 15, expected, integrators, checks, where+" full restored",
           false, true);
   Request(form, state, 2, expected, integrators, checks, where+" before empty");
   Request(form, state, 15, expected, integrators, checks, where+" 0x0",
           true);
}

/// Centered residual differences use free directions and nonconstant states.
/// Mult zeroes essential rows, whereas the gradient has identity diagonals;
/// therefore constrained entries of every perturbation must be zero.
void FiniteDifferences(BlockNonlinearForm &form, const EssentialDofs &essential,
                       Checks &checks, const std::string &where)
{
   const Array<int> &offsets = form.GetBlockTrueOffsets();
   BlockVector state(offsets), direction(offsets), plus(offsets), minus(offsets);
   BlockVector fplus(offsets), fminus(offsets), action(offsets), derivative(offsets);
   for (int i = 0; i < state.Size(); ++i)
   { state(i) = 0.8 + 0.15*std::sin(0.47*(i+1)); }
   state.SyncToBlocks();
   for (int col = 0; col < 2; ++col)
   {
      direction = 0.0;
      for (int i = 0; i < direction.GetBlock(col).Size(); ++i)
      { direction.GetBlock(col)(i) = std::cos(0.31*(i+1)); }
      direction.GetBlock(col).SetSubVector(essential[col], 0.0);
      direction.SyncFromBlocks();
      if (direction.Norml2() == 0.0) { continue; }
      for (real_t h : { real_t(1e-4), real_t(2e-6) })
      {
         plus = state;
         minus = state;
         plus.Add(h, direction);
         minus.Add(-h, direction);
         form.Mult(plus, fplus);
         form.Mult(minus, fminus);
         derivative = fplus;
         derivative -= fminus;
         derivative /= 2*h;
         derivative.SyncToBlocks();
         for (int row = 0; row < 2; ++row)
         {
            // Only this block is available: residual differences still need
            // the other field's complete state, including its essential DOFs.
            Array2D<int> mask = Mask(1 << (2*row+col));
            form.GetGradient(state, mask).Mult(direction, action);
            action.SyncToBlocks();
            Vector error(action.GetBlock(row));
            error -= derivative.GetBlock(row);
            const real_t reference = derivative.GetBlock(row).Norml2();
            ++checks.differences;
            checks.Require(std::isfinite(error.Norml2()) &&
                           error.Norml2() <= 2e-8 * (1.0 + reference),
                           where + " residual FD row=" + std::to_string(row) +
                           " col=" + std::to_string(col) +
                           " h=" + std::to_string(h));
         }
      }
   }
}

/// Confirm the hint owns its copy and treats any nonzero integer as enabled.
void HintContract(Checks &checks)
{
   PolynomialMassIntegrator integrator;
   Array2D<int> mask = Mask(6, true);
   integrator.SetGradientBlockMask(mask);
   mask = 0;
   checks.Require(!integrator.Enabled(0,0) && integrator.Enabled(0,1) &&
                  integrator.Enabled(1,0) && !integrator.Enabled(1,1),
                  "integrator copied signed mask");
   integrator.SetGradientBlockMask(Array2D<int>());
   checks.Require(integrator.Enabled(0,0) && integrator.Enabled(0,1) &&
                  integrator.Enabled(1,0) && integrator.Enabled(1,1),
                  "empty integrator hint enables all blocks");
}

/// Run analytic, finite-difference, legacy, stale-local, and BC regressions.
void DomainCases(bool nonconforming, Checks &checks)
{
   Mesh mesh = Mesh::MakeCartesian2D(3, 2, Element::QUADRILATERAL,
                                     true, 1.5, 1.0);
   if (nonconforming)
   {
      mesh.EnsureNCMesh();
      Array<int> refine(1);
      refine[0] = 0;
      mesh.GeneralRefinement(refine, 1);
   }
   H1_FECollection q1(1, 2), q2(2, 2);
   FiniteElementSpace first(&mesh, &q1), second(&mesh, &q2);
   Array<FiniteElementSpace *> spaces(2);
   spaces[0] = &first;
   spaces[1] = &second;
   checks.Require(first.GetTrueVSize() != second.GetTrueVSize(),
                  "unequal Q1/Q2 space dimensions");
   if (nonconforming)
   {
      checks.Require(first.GetVSize() > first.GetTrueVSize() &&
                     second.GetVSize() > second.GetTrueVSize(),
                     "nonconforming case actually has constrained DOFs");
   }
   const Blocks mass = ExactMass(spaces);
   // Integral of each partition-of-unity product equals the rectangle area.
   for (const auto &block : mass)
   {
      real_t sum = 0.0;
      for (int i = 0; i < block.Height(); ++i)
      { for (int j = 0; j < block.Width(); ++j) { sum += block(i,j); } }
      checks.Require(std::abs(sum-1.5) < 2e-12, "exact mass partition of unity");
   }
   for (int variant = 0; variant < 3; ++variant)
   {
      BlockNonlinearForm form(spaces);
      std::vector<TrackedIntegrator *> integrators;
      // Variant 2 deliberately puts legacy matrices into the shared local
      // array before the aware integrator leaves off-mask entries untouched.
      if (variant == 2)
      {
         auto *legacy = new PolynomialMassIntegrator(false);
         form.AddDomainIntegrator(legacy);
         integrators.push_back(legacy);
      }
      auto *integrator = new PolynomialMassIntegrator(variant != 1);
      form.AddDomainIntegrator(integrator);
      integrators.push_back(integrator);
      BlockVector state(form.GetBlockTrueOffsets());
      const std::string prefix = std::string(nonconforming ? "NC" : "conforming") +
                                 " variant=" + std::to_string(variant);
      // Reconfigure BCs on the same form, including recovery from all DOFs fixed.
      for (int mode : {0, 1, 2, 3, 0})
      {
         ++checks.cases;
         const EssentialDofs essential = SetBC(form, mode);
         ConstantState(state, 1.1, 0.6);
         const Blocks expected = Oracle(mass, 1.1, 0.6, essential,
                                        variant == 2 ? 2 : 1);
         const std::string where = prefix + " BC=" + std::to_string(mode);
         // Full assembly also has to pass the independent analytic oracle.
         const Blocks full = Snapshot(
            dynamic_cast<BlockOperator &>(form.GetGradient(state)), checks, where);
         for (int b = 0; b < 4; ++b)
         { checks.Near(full[b], expected[b], where + " analytic full"); }
         ExerciseMasks(form, state, expected, integrators, checks, where);

         // A single requested diagonal block still depends on BOTH fields.
         // Vary the field outside the selected row/column without a full call.
         ConstantState(state, 1.1, 1.3);
         Request(form, state, 1, Oracle(mass, 1.1, 1.3, essential,
                                       variant == 2 ? 2 : 1),
                 integrators, checks, where+" changed v, block00");
         ConstantState(state, 0.4, 1.3);
         Request(form, state, 8, Oracle(mass, 0.4, 1.3, essential,
                                       variant == 2 ? 2 : 1),
                 integrators, checks, where+" changed u, block11");
         Request(form, state, 15, Oracle(mass, 0.4, 1.3, essential,
                                        variant == 2 ? 2 : 1),
                 integrators, checks, where+" changed state full", false, true);
         FiniteDifferences(form, essential, checks, where);
      }
      if (variant == 2)
      {
         checks.Require(integrator->stale_off_mask > 0,
                        prefix+" exercised stale local matrices");
      }
      if (variant == 0)
      {
         SetBC(form, 0);
         ConstantState(state, 1.1, 0.6);
         const std::size_t before_full = integrator->constructions;
         form.GetGradient(state);
         const std::size_t full_count = integrator->constructions-before_full;
         const std::size_t before_one = integrator->constructions;
         form.GetGradient(state, Mask(1));
         const std::size_t one_count = integrator->constructions-before_one;
         checks.Require(full_count == std::size_t(4*mesh.GetNE()) &&
                        one_count == std::size_t(mesh.GetNE()),
                        prefix+" benchmark 4 versus 1 per element");
         std::cout << "COUNTS " << (nonconforming ? "nonconforming" : "conforming")
                   << " elements=" << mesh.GetNE() << " full=" << full_count
                   << " block00=" << one_count
                   << " (local matrices; no speed claim)\n";
      }
   }
}

/// Full-vs-selected and FD checks for boundary elements and both face paths.
void SurfaceCases(int kind, Checks &checks)
{
   Mesh base = Mesh::MakeCartesian2D(3, 3, Element::QUADRILATERAL,
                                     true, 1.0, 1.0);
   std::vector<Vector> translations;
   translations.emplace_back(2);
   translations.emplace_back(2);
   translations[0] = 0.0;
   translations[1] = 0.0;
   translations[0](0) = 1.0;
   translations[1](1) = 1.0;
   Mesh mesh = kind == 2 ?
      Mesh::MakePeriodic(base, base.CreatePeriodicVertexMapping(translations)) :
      Mesh(base);
   if (kind == 2)
   {
      // Upstream gradient assembly dereferences every interior-face query.
      // A periodic mesh must have no null queries before we exercise that code.
      bool all_interior = true;
      for (int f = 0; f < mesh.GetNumFaces(); ++f)
      { all_interior = all_interior && mesh.GetInteriorFaceTransformations(f); }
      checks.Require(all_interior && mesh.GetNumFaces() > 0,
                     "periodic face mesh has no boundary-face null queries");
      if (!all_interior) { return; }
   }
   H1_FECollection q1(1, 2), q2(2, 2);
   FiniteElementSpace first(&mesh, &q1), second(&mesh, &q2);
   Array<FiniteElementSpace *> spaces(2);
   spaces[0] = &first;
   spaces[1] = &second;
   const char *names[] = { "boundary-element", "boundary-face", "periodic-face" };
   for (bool honor : {true, false})
   {
      BlockNonlinearForm form(spaces);
      TrackedIntegrator *integrator;
      if (kind == 0)
      {
         integrator = new PolynomialMassIntegrator(honor);
         form.AddBoundaryIntegrator(integrator);
      }
      else
      {
         integrator = new FaceMassIntegrator(honor);
         if (kind == 1) { form.AddBdrFaceIntegrator(integrator); }
         else { form.AddInteriorFaceIntegrator(integrator); }
      }
      const std::vector<TrackedIntegrator *> integrators = {integrator};
      BlockVector state(form.GetBlockTrueOffsets());
      for (int mode = 0; mode < (kind == 2 ? 1 : 3); ++mode)
      {
         ++checks.cases;
         const EssentialDofs essential = SetBC(form, mode);
         ConstantState(state, 1.1, 0.6);
         const std::string where = std::string(names[kind]) +
                                   (honor ? " aware" : " legacy") +
                                   " BC=" + std::to_string(mode);
         const Blocks full = Snapshot(
            dynamic_cast<BlockOperator &>(form.GetGradient(state)), checks, where);
         if (mode == 0)
         {
            for (const auto &block : full)
            {
               real_t norm = 0.0;
               for (int i = 0; i < block.Height(); ++i)
               { for (int j = 0; j < block.Width(); ++j)
                  { norm += std::abs(block(i,j)); } }
               checks.Require(norm > 1e-6, where + " nonvacuous face block");
            }
         }
         ExerciseMasks(form, state, full, integrators, checks, where);
         FiniteDifferences(form, essential, checks, where);
         checks.Require(integrator->calls > 0, where + " callbacks exercised");
      }
   }
}

/// Run in a subprocess: successful validation aborts/throws before returning.
int InvalidMask(int rows, int cols)
{
   if (rows < 0 || cols < 0 || (rows == 2 && cols == 2) ||
       (rows == 0 && cols == 0))
   {
      std::cerr << "Supply nonnegative dimensions other than 0x0 and 2x2\n";
      return 2;
   }
   Mesh mesh = Mesh::MakeCartesian2D(1, 1, Element::QUADRILATERAL);
   H1_FECollection collection(1, 2);
   FiniteElementSpace first(&mesh, &collection), second(&mesh, &collection);
   Array<FiniteElementSpace *> spaces(2);
   spaces[0] = &first;
   spaces[1] = &second;
   BlockNonlinearForm form(spaces);
   form.AddDomainIntegrator(new PolynomialMassIntegrator);
   BlockVector state(form.GetBlockTrueOffsets());
   ConstantState(state, 1.1, 0.6);
   Array2D<int> mask(rows, cols);
   mask = 1;
   std::cerr << "Expect MFEM rejection of gradient mask "
             << rows << 'x' << cols << '\n';
   form.GetGradient(state, mask);
   // The runner should check for MFEM's validation diagnostic / abort.
   // An ordinary assertion failure here is not successful rejection by MFEM.
   std::cerr << "FAIL: invalid gradient mask was accepted\n";
   return 1;
}

} // namespace

int main(int argc, char **argv)
{
   if (argc == 4 && std::string(argv[1]) == "--invalid-mask")
   {
      int rows, cols;
      try
      {
         std::size_t rlen, clen;
         rows = std::stoi(argv[2], &rlen);
         cols = std::stoi(argv[3], &clen);
         if (argv[2][rlen] != '\0' || argv[3][clen] != '\0')
         {
            std::cerr << "Mask dimensions must be integers\n";
            return 2;
         }
      }
      catch (const std::exception &)
      {
         std::cerr << "Mask dimensions must be integers\n";
         return 2;
      }
      return InvalidMask(rows, cols);
   }
   if (argc != 1)
   {
      std::cerr << "Usage: " << argv[0]
                << " [--invalid-mask ROWS COLS]\n";
      return 2;
   }
   Checks checks;
   try
   {
      HintContract(checks);
      DomainCases(false, checks);
      DomainCases(true, checks);
      SurfaceCases(0, checks);
      SurfaceCases(1, checks);
      SurfaceCases(2, checks);
   }
   catch (const std::exception &exception)
   {
      checks.Require(false, std::string("unexpected exception: ") + exception.what());
   }
   std::cout << (checks.failures ? "FAIL" : "PASS")
             << " cases=" << checks.cases << " mask_checks=" << checks.masks
             << " residual_FD=" << checks.differences
             << " assertions=" << checks.assertions
             << " failures=" << checks.failures << '\n';
   return checks.failures ? 1 : 0;
}
