#include "Modules/VQE.hpp"
#include "Angular/Angular.hpp"
#include "Coulomb/Coulomb.hpp"
#include "IO/InputBlock.hpp"
#include "LinAlg/Matrix.hpp"
#include "MBPT/CorrelationPotential.hpp"
#include "Physics/AtomData.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/format.hpp"
#include "fmt/ostream.hpp"
#include "qip/Vector.hpp"
#include <array>
#include <fstream>
#include <vector>

namespace Module {

using nkIndex = DiracSpinor::Index;

//==============================================================================
//==============================================================================
//==============================================================================
// This is the actual module that runs:
void VQE(const IO::InputBlock &input, const Wavefunction &wf) {

  // Check input options:
  input.check(
      {{"frozen_core", "Core states that are not included in CI expansion. By "
                       "default, this is the same as the HartreeFock{} core"},
       {"basis", "Basis used for CI expansion; must be a sub-set of MBPT basis "
                 "[default: 5spd]"},
       {"J", "List of J angular symmetry for CSFs (comma separated). For "
             "half-integer, enter as floats: '0.5' not '1/2' [default: 0]"},
       {"num_solutions", "Number of CI solutions to find (for each J/pi) [5]"},
       {"e0", "Optional: ground-state energy (in 1/cm) for relative energies. "
              "If not given, will assume lowest J+"},
       {"write_integrals", "Writes CSFs, CI matrix, and 1 and 2 particle "
                           "integrals to plain file [false]"}});
  // If we are just requesting 'help', don't run module:
  if (input.has_option("help")) {
    return;
  }

  // Decide if should write single-particle integrals to file:
  const auto write_integrals = input.get("write_integrals", false);

  //----------------------------------------------------------------------------
  // Single-particle basis:
  std::cout << "\nConstruct single-particle basis:\n";

  // Determine the sub-set of basis to use in CI:
  const auto basis_string = input.get("basis", std::string{"5spd"});

  const auto frozen_core_string =
      input.get("frozen_core", wf.coreConfiguration());

  // Select from wf.basis() [MBPT basis], those which match input 'basis_string'
  const std::vector<DiracSpinor> ci_sp_basis =
      // basis_subset(wf.basis(), basis_string, wf.FermiLevel());
      basis_subset(wf.basis(), basis_string, frozen_core_string);

  // Print info re: basis to screen:
  std::cout << "\nUsing " << DiracSpinor::state_config(ci_sp_basis) << " = "
            << ci_sp_basis.size() << " relativistic single-particle orbitals\n";

  //----------------------------------------------------------------------------

  std::cout << "\nSingle-particle matrix elements:\n";

  // Correlation potential (one-particle Sigma matrix):
  const auto Sigma = wf.Sigma();

  // Create lookup table for one-particle matrix elements, h1
  Coulomb::meTable h1;

  // print all single-body integrals to file:
  std::string one_file = "h1.txt";
  std::ofstream h1_file;
  if (write_integrals) {
    h1_file.open(one_file);
    std::cout << "Writing single-particle matrix elements to file: " << one_file
              << "\n";
    h1_file << "# a  b  i_a  i_b  <a|h_1|b>     ## (i is 'index' of state)\n";
  }
  // Calculate + store all 1-body integrals; optionally write to file
  for (const auto &v : ci_sp_basis) {
    for (const auto &w : ci_sp_basis) {
      if (w > v)
        continue;
      const auto h0_vw = v == w ? v.en() : 0.0;
      // const auto h0_vw = wf.Hab(v, w);
      const auto Sigma_vw = Sigma ? v * Sigma->SigmaFv(w) : 0.0;
      h1.add(v, w, h0_vw + Sigma_vw);
      if (v != w)
        h1.add(w, v, h0_vw + Sigma_vw);
      if (write_integrals) {
        fmt::print(h1_file, "{} {} {} {} {:+.9e}\n", v.shortSymbol(),
                   w.shortSymbol(), v.nk_index(), w.nk_index(),
                   h0_vw + Sigma_vw);
      }
    }
  }

  //----------------------------------------------------------------------------

  std::cout << "\nCalculate two-body Coulomb integrals: W^k_abcd\n";

  // Lookup table; stores all qk's
  Coulomb::QkTable qk;
  const auto qk_filename =
      wf.identity() + DiracSpinor::state_config(ci_sp_basis) + ".qk";
  // Try to read from disk (may already have calculated Qk)
  const auto read_from_file_ok = qk.read(qk_filename);
  if (!read_from_file_ok) {
    // if didn't find Qk file to read in, calculate from scratch:
    const Coulomb::YkTable yk(ci_sp_basis);
    qk.fill(ci_sp_basis, yk);
    qk.write(qk_filename);
  }

  // Writes Rk integrals to text file
  if (write_integrals) {
    write_CoulombIntegrals(ci_sp_basis, qk);
  }

  //----------------------------------------------------------------------------
  const auto J_list = input.get("J", std::vector<double>{0.0});
  const auto num_solutions = input.get("num_solutions", 5);
  double e0 = input.get("e0", 0.0) / PhysConst::Hartree_invcm;
  // even parity:
  for (auto &J : J_list) {
    auto e1 = run_CI(wf.atomicSymbol(), ci_sp_basis, int(std::round(2 * J)), +1,
                     num_solutions, h1, qk, e0, write_integrals);
    if (e0 == 0.0)
      e0 = e1;
  }
  // odd parity:
  for (auto &J : J_list) {
    run_CI(wf.atomicSymbol(), ci_sp_basis, int(std::round(2 * J)), -1,
           num_solutions, h1, qk, e0, write_integrals);
  }
}

//==============================================================================
//==============================================================================

//==============================================================================
//! Very basic two-electron CSF.
class CSF2 {
public:
  // nb: array of states is always sorted
  std::array<const DiracSpinor *, 2> states;

  CSF2(const DiracSpinor &a, const DiracSpinor &b)
      : states(a <= b ? std::array{&a, &b} : std::array{&b, &a}) {}

  const DiracSpinor *state(std::size_t i) const { return states.at(i); }

  friend bool operator==(const CSF2 &A, const CSF2 &B) {
    // only works because states is sorted
    return A.states == B.states;
  }
  friend bool operator!=(const CSF2 &A, const CSF2 &B) {
    // only works because states is sorted
    return !(A.states == B.states);
  }

  //! Returns number of different orbitals between two CSFs
  static nkIndex num_different(const CSF2 &A, const CSF2 &B) {
    if (A == B)
      return 0;
    if (A.state(0) == B.state(0) || A.state(1) == B.state(1) || //
        A.state(0) == B.state(1) || A.state(1) == B.state(0))
      return 1;
    return 2;
  }

  //! returns _different_ orbitals, for case where CSFs differ by 1.
  //! i.e., returns {n,a} where |A> = |B_a^n> (i.e., A has n, but not a)
  static std::array<const DiracSpinor *, 2> diff_1_na(const CSF2 &A,
                                                      const CSF2 &B) {
    assert(num_different(A, B) == 1); // only valid in this case
    if (A.state(0) == B.state(0))
      return {A.state(1), B.state(1)};
    if (A.state(1) == B.state(1))
      return {A.state(0), B.state(0)};
    if (A.state(0) == B.state(1))
      return {A.state(1), B.state(0)};
    if (A.state(1) == B.state(0))
      return {A.state(0), B.state(1)};
    assert(false); // should be unreachable, for testing
  }
};

//==============================================================================
// Takes a subset of input basis according to subset_string.
// Only states *not* included in frozen_core_string are included.
std::vector<DiracSpinor> basis_subset(const std::vector<DiracSpinor> &basis,
                                      const std::string &subset_string,
                                      const std::string &frozen_core_string) {

  // Form 'subset' from {a} in 'basis', if:
  //    a _is_ in subset_string AND
  //    a _is not_ in basis string

  std::vector<DiracSpinor> subset;
  const auto nmaxk_list = AtomData::n_kappa_list(subset_string);
  const auto core_list = AtomData::core_parser(frozen_core_string);

  for (const auto &a : basis) {

    // Check if a is present in 'subset_string'
    const auto nk =
        std::find_if(nmaxk_list.cbegin(), nmaxk_list.cend(),
                     [&a](const auto &tnk) { return a.kappa() == tnk.second; });
    if (nk == nmaxk_list.cend())
      continue;
    // nk is now max n, for given kappa {max_n, kappa}
    if (a.n() > nk->first)
      continue;

    // assume only filled shells in frozen core
    const auto core = std::find_if(
        core_list.cbegin(), core_list.cend(), [&a](const auto &tcore) {
          return a.n() == tcore.n && a.l() == tcore.l;
        });

    if (core != core_list.cend())
      continue;
    subset.push_back(a);
  }
  return subset;
}

//==============================================================================
void write_CSFs(const std::vector<CSF2> &CSFs, int twoJ,
                const std::string &csf_fname) {

  std::cout << "Writing CSFs and projections to files: {csf/proj}-" << csf_fname
            << "\n";
  std::ofstream csf_file("csf-" + csf_fname);
  std::ofstream proj_file("proj-" + csf_fname);

  csf_file << "# csf_index  a  b  i_a  i_b \n";
  proj_file << "# proj_index  a  b  i_a  i_b  2*ma  2*mb  CGC\n";
  int csf_count = 0;
  int proj_count = 0;
  for (const auto &csf : CSFs) {

    const auto &v = *csf.state(0);
    const auto &w = *csf.state(1);

    fmt::print(csf_file, "{} {} {} {} {}\n", proj_count, v.shortSymbol(),
               w.shortSymbol(), v.nk_index(), w.nk_index());
    ++csf_count;

    // Each individual m projection:
    for (int two_m_v = -v.twoj(); two_m_v <= v.twoj(); two_m_v += 2) {
      const auto two_m_w = twoJ - two_m_v;
      if (std::abs(two_m_w) > w.twoj())
        continue;
      const auto cgc =
          Angular::cg_2(v.twoj(), two_m_v, w.twoj(), two_m_w, twoJ, twoJ);
      if (cgc == 0.0)
        continue;

      fmt::print(proj_file, "{} {} {} {} {} {} {} {:.6f}\n", proj_count,
                 v.shortSymbol(), w.shortSymbol(), v.nk_index(), w.nk_index(),
                 two_m_v, two_m_w, cgc);
      ++proj_count;
    }
    proj_file << "\n";
  }
}

//==============================================================================
void write_CoulombIntegrals(const std::vector<DiracSpinor> &ci_sp_basis,
                            const Coulomb::QkTable &qk) {

  std::string two_file_R = "h2_Rk.txt";

  std::cout << "\nWriting radial Coulomb intgrals 'R^k' to file: " << two_file_R
            << "\n";
  // print all two-particle integrals Rk to file:
  std::ofstream Rk_file(two_file_R);
  Rk_file << "# a  b  c  d  i_a  i_b  i_c  i_d  k  R^k_abcd  Q^k_abcd     ## "
             "(k is multipolarity)\n";

  for (const auto &a : ci_sp_basis) {
    for (const auto &b : ci_sp_basis) {
      for (const auto &c : ci_sp_basis) {
        for (const auto &d : ci_sp_basis) {

          // only print unique integrals:
          if (!qk.is_NormalOrdered(a, b, c, d))
            continue;

          const auto [kmin, kmax] = Coulomb::k_minmax_Q(a, b, c, d);
          // can skip every 2nd k: parity
          for (int k = kmin; k <= kmax; k += 2) {
            const auto rk = qk.R(k, a, b, c, d);
            const auto Qk = qk.Q(k, a, b, c, d);
            fmt::print(Rk_file, "{} {} {} {} {} {} {} {} {} {:+.9e} {:+.9e}\n",
                       a.shortSymbol(), b.shortSymbol(), c.shortSymbol(),
                       d.shortSymbol(), a.nk_index(), b.nk_index(),
                       c.nk_index(), d.nk_index(), k, rk, Qk);
            // if required, could output g as well
          }
        }
      }
    }
  }
}

//==============================================================================
// Forms list of 2-particle CSFs with given symmetry
std::vector<CSF2> form_CSFs(int twoJ, int parity,
                            const std::vector<DiracSpinor> &ci_sp_basis) {

  std::vector<CSF2> CSFs;

  for (const auto &v : ci_sp_basis) {
    for (const auto &w : ci_sp_basis) {

      // Symmetric: only include unique CSFs once
      if (w < v)
        continue;

      // Parity symmetry:
      if (v.parity() * w.parity() != parity)
        continue;

      // J triangle rule (use M=Jz=J):
      if (v.twoj() + w.twoj() < twoJ || std::abs(v.twoj() - w.twoj()) > twoJ)
        continue;

      // identical particles can only give even J (cannot have mv=mw)
      if (v == w && twoJ % 4 != 0)
        continue;

      CSFs.emplace_back(v, w);
    }
  }

  return CSFs;
}

//==============================================================================
double CSF2_Coulomb(const Coulomb::QkTable &qk, const DiracSpinor &a,
                    const DiracSpinor &b, const DiracSpinor &c,
                    const DiracSpinor &d, int twoJ) {

  // If c==d, or a==b : can make short-cut due to symmetry
  // More efficient to use two Q's than W:

  double out = 0.0;

  // Direct part:
  const auto [k0, k1] = Coulomb::k_minmax_Q(a, b, c, d);
  for (int k = k0; k <= k1; k += 2) {
    const auto sjs =
        Angular::sixj_2(a.twoj(), b.twoj(), twoJ, d.twoj(), c.twoj(), 2 * k);
    if (sjs == 0.0)
      continue;
    const auto qk_abcd = qk.Q(k, a, b, c, d);
    const auto s = Angular::neg1pow_2(a.twoj() + c.twoj() + 2 * k + twoJ);
    out += s * sjs * qk_abcd;
  }

  // Take advantage of symmetries: faster (+ numerically stable)
  // c == d => J is even (identical states), eta2=1/sqrt(2)
  // eta_ab = 1/sqrt(2) if a==b
  // Therefore: e.g., if c==d
  // => eta_ab * eta_cd * (out + (-1)^J*out) = eta_ab * sqrt(2) * out
  if (a == b && c == d) {
    return out;
  } else if (c == d || a == b) {
    // by {ab},{cd} symmetry: same works for case a==b
    return std::sqrt(2.0) * out;
  }

  // Exchange part:
  const auto [l0, l1] = Coulomb::k_minmax_Q(a, b, d, c);
  for (int k = l0; k <= l1; k += 2) {
    const auto sjs =
        Angular::sixj_2(a.twoj(), b.twoj(), twoJ, c.twoj(), d.twoj(), 2 * k);
    if (sjs == 0.0)
      continue;
    const auto qk_abdc = qk.Q(k, a, b, d, c);
    const auto s = Angular::neg1pow_2(a.twoj() + c.twoj() + 2 * k);
    out += s * sjs * qk_abdc;
  }

  return out;
}

//==============================================================================
// Determines CI Hamiltonian matrix element for two 2-particle CSFs, a and b
double Hab(const CSF2 &A, const CSF2 &B, int twoJ,
           const Coulomb::meTable<double> &h1, const Coulomb::QkTable &qk) {

  // Calculates matrix element of the CI Hamiltonian between two CSFs

  // This doesn't work with Sigma, but is good test for Sigma=0 case
  // const auto [v, w] = A.states;
  // const auto [x, y] = B.states;
  // const auto Evw = (v == x && y == w) ? v->en() + w->en() : 0.0;
  // return Evw + CSF2_Coulomb(qk, *v, *w, *x, *y, twoJ);

  const auto num_different = CSF2::num_different(A, B);
  if (num_different == 0) {
    return Hab_0(A, B, twoJ, h1, qk);
  }
  if (num_different == 1) {
    return Hab_1(A, B, twoJ, h1, qk);
  }
  if (num_different == 2) {
    return Hab_2(A, B, twoJ, h1, qk);
  }
  assert(false); // can't have more than 2 different in a 2-particle CSF!
  return 0.0;
}

//==============================================================================
// CI Hamiltonian matrix element for two 2-particle CSFs: diagonal case
double Hab_0(const CSF2 &A, const CSF2 &, int twoJ,
             const Coulomb::meTable<double> &h1, const Coulomb::QkTable &qk) {

  // Orbitals:
  const auto &a = *A.state(0);
  const auto &b = *A.state(1);

  // lookup single-particle matrix elements (includes Sigma)
  const auto h1_aa = h1.getv(a, a);
  const auto h1_bb = h1.getv(b, b);

  return h1_aa + h1_bb + CSF2_Coulomb(qk, a, b, a, b, twoJ);
}

//==============================================================================
// CI Hamiltonian matrix element for two 2-particle CSFs: differ by 1 case
double Hab_1(const CSF2 &A, const CSF2 &B, int twoJ,
             const Coulomb::meTable<double> &h1, const Coulomb::QkTable &qk) {

  // get the 'different' orbitals in A/B:
  const auto [n, a] = CSF2::diff_1_na(A, B);

  // one-electron part (includes Sigma):
  const auto h1_na = h1.getv(*n, *a);

  const auto [v, w] = A.states;
  const auto [x, y] = B.states;

  return h1_na + CSF2_Coulomb(qk, *v, *w, *x, *y, twoJ);
}

//==============================================================================
// CI Hamiltonian matrix element for two 2-particle CSFs: differ by 2 case
double Hab_2(const CSF2 &A, const CSF2 &B, int twoJ,
             const Coulomb::meTable<double> &, const Coulomb::QkTable &qk) {

  const auto [a, b] = A.states;
  const auto [n, m] = B.states;
  return CSF2_Coulomb(qk, *a, *b, *n, *m, twoJ);
}

//==============================================================================
double run_CI(const std::string &atom_name,
              const std::vector<DiracSpinor> &ci_sp_basis, int twoJ, int parity,
              int num_solutions, const Coulomb::meTable<double> &h1,
              const Coulomb::QkTable &qk, double e0, bool write_integrals) {
  //----------------------------------------------------------------------------

  auto printJ = [](int twoj) {
    return twoj % 2 == 0 ? std::to_string(twoj / 2) :
                           std::to_string(twoj) + "/2";
  };
  auto printPi = [](int pi) { return pi > 0 ? "even" : "odd"; };

  fmt::print("\nForm CSFs for J={}, {} parity\n", printJ(twoJ),
             printPi(parity));
  std::cout << std::flush;

  if (twoJ < 0) {
    std::cout << "twoJ must >=0\n";
    return 0.0;
  }
  if (twoJ % 2 != 0) {
    std::cout << "twoJ must be even for two-electron CSF\n";
  }

  std::vector<CSF2> CSFs = form_CSFs(twoJ, parity, ci_sp_basis);
  std::cout << "Total CSFs: " << CSFs.size() << "\n";
  std::cout << std::flush;

  //----------------------------------------------------------------------------

  std::string output_prefix =
      atom_name + "_" + printJ(twoJ) + "_" + printPi(parity);

  // Write CSFs (just labels) to file:
  if (write_integrals)
    write_CSFs(CSFs, twoJ, output_prefix + ".txt");

  //----------------------------------------------------------------------------
  fmt::print("Construct CI matrix for J={}, {} parity:\n", printJ(twoJ),
             printPi(parity));
  std::cout << std::flush;

  LinAlg::Matrix Hci(CSFs.size(), CSFs.size());

  {
    IO::ChronoTimer t("Fill matrix");
#pragma omp parallel for collapse(2)
    for (std::size_t iA = 0; iA < CSFs.size(); ++iA) {
      // go to iB <= iA only: symmetric matrix
      for (std::size_t iB = 0; iB <= iA; ++iB) {
        const auto &A = CSFs.at(iA);
        const auto &B = CSFs.at(iB);

        const auto E_AB = Hab(A, B, twoJ, h1, qk);
        Hci(iA, iB) = E_AB;
        // fill other half of symmetric matrix:
        if (iB != iA) {
          Hci(iB, iA) = E_AB;
        }
      }
    }
  }
  std::cout << std::flush;

  // Write CI matrix (H matrix) to file
  if (write_integrals) {
    std::string ci_fname = "ci-" + output_prefix + ".txt";
    std::cout << "Writing CI matrix to file: " << ci_fname << "\n";
    std::ofstream ci_file(ci_fname);
    ci_file << "# in matrix/table form: \n";
    for (std::size_t iA = 0; iA < CSFs.size(); ++iA) {
      for (std::size_t iX = 0; iX < CSFs.size(); ++iX) {
        fmt::print(ci_file, "{:+.6e} ", Hci(iA, iX));
      }
      ci_file << "\n";
    }
  }

  //----------------------------------------------------------------------------
  std::cout << std::flush;

  const auto [val, vec] = LinAlg::symmhEigensystem(Hci, true);
  const auto E0 = e0 == 0.0 ? val(0) : e0;

  fmt::print("Full CI for J={}, pi={} : E0 = {:.1f} cm^-1\n\n", printJ(twoJ),
             printPi(parity), E0 * PhysConst::Hartree_invcm);
  std::cout << std::flush;

  for (std::size_t i = 0; i < val.size() && int(i) < num_solutions; ++i) {

    fmt::print(
        "{:<2} {} {:+1}  {:+11.8f} au  {:+11.2f} cm^-1  {:11.2f} cm^-1\n", i,
        0.5 * twoJ, parity, val(i), val(i) * PhysConst::Hartree_invcm,
        (val(i) - E0) * PhysConst::Hartree_invcm);

    for (std::size_t j = 0ul; j < vec.cols(); ++j) {
      const auto cj = 100.0 * std::pow(vec(i, j), 2);
      if (cj > 1.0) {
        fmt::print("  {:>4s},{:<4s} {:5.3f}%\n",
                   CSFs.at(j).state(0)->shortSymbol(),
                   CSFs.at(j).state(1)->shortSymbol(), cj);
      }
    }
    std::cout << "\n";
  }

  //----------------------------------------------------------------------------
  std::cout
      << "\n`Direct' energy calculation: E = Σ_{IJ} c_I * c_J * <I|H|J>:\n";
  std::cout
      << "(Using the CI expansion coefficients from full CI, just a test)\n";
  // Energy calculation for ground state:
  double E_direct1 = 0.0, E_direct2 = 0.0;
  const auto Nci = vec.rows(); // number of CSFs
  // Energy:  E = Sum_ij c_i * c_j * <i|H|j>
  for (std::size_t i = 0ul; i < Nci; ++i) {
    const auto &csf_i = CSFs.at(i); // the ith CSF
    const auto ci = vec.at(0, i);   // the ith CI coefficient (for 0th e.val)
    for (std::size_t j = 0ul; j < Nci; ++j) {
      const auto &csf_j = CSFs.at(j); // jth CSF
      const auto cj = vec.at(0, j);   // the jth CI coefficient (for 0th e.val)
      // use pre-calculated CI matrix:
      E_direct1 += ci * cj * Hci.at(i, j);
      // Calculate MEs on-the-fly
      E_direct2 += ci * cj * Hab(csf_i, csf_j, twoJ, h1, qk);
    }
  }

  std::cout << "E0 = " << val.at(0) * PhysConst::Hartree_invcm
            << " cm^-1  (from diagonalisation)\n";
  std::cout << "E0 = " << E_direct1 * PhysConst::Hartree_invcm
            << " cm^-1  (uses pre-calculated CI matrix)\n";
  std::cout << "E0 = " << E_direct2 * PhysConst::Hartree_invcm
            << " cm^-1  (calculates H matrix elements from scratch)\n";

  return val.at(0);
}

} // namespace Module
