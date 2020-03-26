#include "Wavefunction/Wavefunction.hpp"
#include "DiracODE/DiracODE.hpp"
#include "MBPT/CorrelationPotential.hpp"
#include "Maths/Grid.hpp"
#include "Maths/Interpolator.hpp"
#include "Maths/NumCalc_quadIntegrate.hpp"
#include "Physics/AtomData.hpp"
#include "Physics/NuclearPotentials.hpp"
#include "Physics/PhysConst_constants.hpp"
#include "Physics/RadiativePotential.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include <algorithm> //for sort
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

//******************************************************************************
void Wavefunction::solveDirac(DiracSpinor &psi, double e_a,
                              const std::vector<double> &vex,
                              int log_dele_or) const
// Uses Adams::boundState to solve Dirac Eqn for local potential (Vnuc + Vdir)
// If no e_a is given, will use the existing one!
// (Usually, a better guess should be given, using P.T.)
// Note: optionally takes in exchange potential! (see overloaded above)
// Note: Uses the "dodgy" re-scaled exchange potenital:
// Vex\psi_a = sum_b vex_a psi_b -> [sum_b vex_a (psi_b/psi_a)] psi_a
// so, here, vex = [sum_b vex_a (psi_b/psi_a)]
// This is not ideal..
{
  const auto v_a = NumCalc::add_vectors(get_Vlocal(psi.l()), vex);
  if (e_a != 0) {
    psi.en = e_a;
  } else if (psi.en == 0) {
    psi.en = enGuessVal(psi.n, psi.k);
  }
  DiracODE::boundState(psi, psi.en, v_a, get_Hmag(psi.l()), m_alpha,
                       log_dele_or);
}

//------------------------------------------------------------------------------
void Wavefunction::solveDirac(DiracSpinor &psi, double e_a,
                              int log_dele_or) const
// Overloaded version; see above
// This one doesn't have exchange potential
{
  // std::vector<double> empty_vec;
  return solveDirac(psi, e_a, {}, log_dele_or);
}

//******************************************************************************
void Wavefunction::determineCore(std::string str_core_in)
// Takes in a string list for the core configuration, outputs an int list
// Takes in previous closed shell (noble), + 'rest' (or just the rest)
// E.g:
//   Core of Cs: Xe
//   Core of Gold: Xe 4f14 5d10
// 'rest' is in form nLm : n=n, L=l, m=number of electrons in that nl shell.
{

  // Check if integer; if so, V^N-M, '-M' is input integer.
  // Use 'guess' for core
  auto first_char = str_core_in.substr(0, 1);
  if (first_char == "0" || first_char == "-") {
    try {
      auto m = std::stoi(str_core_in);
      str_core_in = AtomData::guessCoreConfigStr(m_Z + m);
    } catch (...) {
    }
  }

  m_core_configs = AtomData::core_parser(str_core_in);

  bool bad_core = false;
  m_core_string = "";
  for (auto config : m_core_configs) {
    if (!config.ok()) {
      m_core_string += " **";
      bad_core = true;
    }
    m_core_string += config.symbol();
    if (config != m_core_configs.back())
      m_core_string += ",";
  }

  if (bad_core) {
    std::cout << "Problem with core: " << str_core_in << " = \n";
    std::cout << m_core_string << "\n";
    std::cout << "In this house, we obey the Pauli exclusion principle!\n";
    std::abort();
  }

  // Count number of electrons in the core
  num_core_electrons = 0;
  for (const auto &config : m_core_configs) {
    num_core_electrons += config.num;
  }

  if (num_core_electrons > m_Z) {
    std::cout << "Problem with core: " << str_core_in << "\n";
    std::cout << "= " << m_core_string << "\n";
    std::cout << "= " << AtomData::niceCoreOutput(m_core_string) << "\n";
    std::cout << "Too many electrons: N_core=" << num_core_electrons
              << ", Z=" << m_Z << "\n";
    std::abort();
  }

  return;
}

//******************************************************************************
void Wavefunction::hartreeFockCore(HF::Method method,
                                   const std::string &in_core, double eps_HF,
                                   double h_d, double g_t) {
  // XXX Update this (and HF) so that it doesn't re-Create m_pHF
  // AND, so that can re-run core!
  m_pHF = std::make_unique<HF::HartreeFock>(
      HF::HartreeFock(method, *this, in_core, eps_HF, h_d, g_t));
}

//******************************************************************************
auto Wavefunction::coreEnergyHF() const {
  if (!m_pHF) {
    std::cerr << "WARNING 62: Cant call coreEnergyHF before hartreeFockCore\n";
    return 0.0;
  }
  return m_pHF->calculateCoreEnergy();
}

//******************************************************************************
void Wavefunction::hartreeFockValence(const std::string &in_valence_str) {
  if (!m_pHF) {
    std::cerr << "WARNING 62: Cant call hartreeFockValence before "
                 "hartreeFockCore\n";
    return;
  }
  auto val_lst = AtomData::listOfStates_nk(in_valence_str);
  for (const auto &nk : val_lst) {
    if (!isInCore(nk.n, nk.k) && !isInValence(nk.n, nk.k))
      valence_orbitals.emplace_back(DiracSpinor{nk.n, nk.k, rgrid});
  }
  m_pHF->solveValence();
}

//******************************************************************************
void Wavefunction::hartreeFockBrueckner(int n_min_core) {
  if (!m_pHF) {
    std::cerr << "WARNING 62: Cant call hartreeFockValence before "
                 "hartreeFockCore\n";
    return;
  }
  if (basis.empty() || valence_orbitals.empty() || core_orbitals.empty()) {
    std::cout << "Can't do Brueckner. Check Splines/Valence/Core states?\n";
    return;
  }
  m_pHF->solveBrueckner(basis, n_min_core);
}

//******************************************************************************
void Wavefunction::SOEnergyShift(int n_min_core) {

  std::cout << "\nMBPT(2): Valence energy shifts.\n";
  std::cout << "Matrix elements <v|Sigma(2)|v>:\n";

  if (basis.empty() || valence_orbitals.empty() || core_orbitals.empty()) {
    std::cout << "Can't do MBPT. Check Splines/Valence/Core states?\n";
    return;
  }

  std::vector<DiracSpinor> core, excited;
  for (const auto &Fb : basis) {
    if (isInCore(Fb) && Fb.n >= n_min_core)
      core.push_back(Fb);
    if (!isInCore(Fb))
      excited.push_back(Fb);
  }
  MBPT::CorrelationPotential Sigma2(core, excited);

  double e0 = 0;
  std::cout << " state |  E(HF)      de=<v|S2|v>      E(HF+2)        E(HF+2) "
               "(cm^-1)\n";
  for (const auto &v : valence_orbitals) {

    auto delta = Sigma2(v, v);
    const auto cm = PhysConst::Hartree_invcm;
    if (e0 == 0)
      e0 = (v.en + delta);

    printf("%7s| %10.7f   %+10.7f  =  %10.7f   = %10.3f   %10.3f\n",
           v.symbol().c_str(), v.en, delta, (v.en + delta), (v.en + delta) * cm,
           (v.en + delta - e0) * cm);
    if (std::abs(delta / v.en) > 0.2)
      std::cout << "      *** Warning: delta too large?\n";
  }
}

//******************************************************************************
void Wavefunction::radiativePotential(double x_simple, double x_Ueh,
                                      double x_SEe_h, double x_SEe_l,
                                      double x_SEm, double rcut,
                                      double scale_rN,
                                      const std::vector<double> &x_spd) {

  if (x_spd.size() == 0)
    return;

  if (x_Ueh > 0 || x_SEe_h > 0 || x_SEe_l > 0 || x_SEm > 0 || x_simple > 0) {
    std::cout << "\nIncluding Ginges/Flambaum QED radiative potential (up to r="
              << rcut << "):\n";
  } else if (std::abs(x_simple) > 0) {
    std::cout << "\nIncluding simple exponential radiative potential\n";
  } else {
    return;
  }

  const auto imax = rgrid.getIndex(rcut);
  const auto rN_rad =
      scale_rN * m_nuc_params.r_rms * std::sqrt(5.0 / 3.0) / PhysConst::aB_fm;

  std::vector<double> Vel_tmp(rgrid.num_points);
  std::vector<double> Hmag_tmp(rgrid.num_points);

  if (std::abs(x_simple) > 0) {
    std::cout << "Forming simple exponential potential (Ak=" << x_simple
              << ")\n";
    for (std::size_t i = 0; i < imax; ++i) {
      Vel_tmp[i] -= x_simple * RadiativePotential::vSimpleExp(
                                   rgrid.r[i], rN_rad, m_Z, m_alpha);
    }
  }

  if (x_Ueh > 0) {
    std::cout << "Forming Uehling potential (scale=" << x_Ueh << ")\n";
#pragma omp parallel for
    for (std::size_t i = 0; i < imax; ++i) {
      const auto r = rgrid.r[i];
      const auto v_Ueh = RadiativePotential::vUehling(r, rN_rad, m_Z, m_alpha);
      Vel_tmp[i] -= x_Ueh * v_Ueh;
    }
  }

  if (x_SEe_h > 0) {
    std::cout << "Forming Self-Energy (electric high-f) potential (scale="
              << x_SEe_h << ")\n";
    // The SE high-f part is very slow. We calculate it every few points only,
    // and then interpolate the intermediate points
    const std::size_t stride = 6;
    const auto tmp_max = imax / stride;
    std::vector<double> x(tmp_max), y(tmp_max);
#pragma omp parallel for
    for (std::size_t i = 0; i < tmp_max; ++i) {
      x[i] = rgrid.r[i * stride];
      y[i] = RadiativePotential::vSEh(x[i], rN_rad, m_Z, m_alpha);
    }
    const auto vec_SE_h = Interpolator::interpolate(x, y, rgrid.r);
    for (std::size_t i = 0; i < imax; i++) {
      Vel_tmp[i] -= x_SEe_h * vec_SE_h[i];
    }
  }

  if (x_SEe_l > 0) {
    std::cout << "Forming Self-Energy (electric low-f) potential (scale="
              << x_SEe_l << ")\n";
#pragma omp parallel for
    for (std::size_t i = 0; i < imax; i++) {
      const auto v_SE_l =
          RadiativePotential::vSEl(rgrid.r[i], rN_rad, m_Z, m_alpha);
      Vel_tmp[i] -= x_SEe_l * v_SE_l;
    }
  }

  if (x_SEm > 0) {
    std::cout << "Forming Self-Energy (magnetic) potential "
              << "(scale=" << x_SEm << ")\n";
#pragma omp parallel for
    for (std::size_t i = 0; i < imax; i++) {
      const auto Hr =
          RadiativePotential::vSE_Hmag(rgrid.r[i], rN_rad, m_Z, m_alpha);
      Hmag_tmp[i] += x_SEm * Hr; // nb: plus!
    }
  }

  const bool print_xl = (x_spd.size() > 1 || x_spd.front() != 1.0);
  int l = 0;
  for (const auto &x_l : x_spd) {
    if (print_xl)
      std::cout << "l=" << l << ", x_l=" << x_l << "\n";
    auto V_l = Vel_tmp; // make a copy..dumb
    auto H_l = Hmag_tmp;
    NumCalc::scaleVec(V_l, x_l);
    NumCalc::scaleVec(H_l, x_l);
    vrad.set_Vel(V_l, l);
    vrad.set_Hmag(H_l, l);
    l++;
  }

  // std::cout << "\n";
}

//******************************************************************************
bool Wavefunction::isInCore(int n, int k) const
// Checks if given state is in the core.
{
  for (auto &phi : core_orbitals) {
    if (n == phi.n && k == phi.k)
      return true;
  }
  return false;
}
bool Wavefunction::isInCore(const DiracSpinor &phi) const {
  return isInCore(phi.n, phi.k);
}
bool Wavefunction::isInValence(int n, int k) const
// Checks if given state is in the valence list.
{
  for (auto &phi : valence_orbitals) {
    if (n == phi.n && k == phi.k)
      return true;
  }
  return false;
}

//******************************************************************************
std::size_t Wavefunction::getStateIndex(int n, int k, bool &is_valence) const {

  is_valence = false;
  for (const auto p_orbs : {&core_orbitals, &valence_orbitals}) {
    std::size_t count = 0;
    for (const auto &phi : *p_orbs) {
      if (n == phi.n && k == phi.k)
        return count;
      ++count;
    }
    is_valence = true;
  }
  // std::cerr << "\nFAIL 290 in WF: Couldn't find state n,k=" << n << "," <<
  // k
  //           << "\n";
  // // std::abort();
  return std::max(core_orbitals.size(), valence_orbitals.size()); // invalid
}
//******************************************************************************
const DiracSpinor *Wavefunction::getState(int n, int k,
                                          bool &is_valence) const {
  auto index = getStateIndex(n, k, is_valence);
  if (is_valence) {
    if (index < valence_orbitals.size())
      return &valence_orbitals[index];
  }
  if (index < core_orbitals.size())
    return &core_orbitals[index];
  return nullptr;
  // XXX return POINTER!
}

//******************************************************************************
int Wavefunction::maxCore_n(int ka) const
// Returns the largest n for states with kappa = ka in the core
// Note: ka is optional input; if none given, will be 0 (& not used)
// (used for energy guesses)
// Note: if you give it l instead of kappa, still works!
{
  int max_n = 0;
  for (const auto &phi : core_orbitals) {
    if (phi.k != ka && ka != 0)
      continue;
    if (phi.n > max_n)
      max_n = phi.n;
  }
  return max_n;
}
//******************************************************************************
int Wavefunction::maxCore_l() const {
  int max_l = 0;
  for (const auto &phi : core_orbitals) {
    if (phi.l() > max_l)
      max_l = phi.l();
  }
  return max_l;
}

//******************************************************************************
void Wavefunction::solveInitialCore(const std::string &str_core,
                                    int log_dele_or)
// Solves the Dirac eqn for each state in the core
// Only for local potential (direct part)
// HF/HartreeFockClass.cpp has routines for Hartree Fock
{
  if (!core_orbitals.empty()) {
    std::cerr << "WARNING 254 in Wavefunction:solveInitialCore: States "
                 "already exist! "
              << core_orbitals.size() << "\n";
    return;
  }

  determineCore(str_core);

  for (const auto &config : m_core_configs) {
    int num = config.num;
    if (num == 0)
      continue;
    int n = config.n;
    int l = config.l;
    double en_a = enGuessCore(n, l);
    int k1 = l; // j = l-1/2
    if (k1 != 0) {
      core_orbitals.emplace_back(DiracSpinor{n, k1, rgrid});
      solveDirac(core_orbitals.back(), en_a, log_dele_or);
      core_orbitals.back().occ_frac = double(num) / (4 * l + 2);
      en_a = 0.95 * core_orbitals.back().en;
      if (en_a > 0)
        en_a = enGuessCore(n, l);
    }
    int k2 = -(l + 1); // j=l+1/2
    core_orbitals.emplace_back(DiracSpinor{n, k2, rgrid});
    solveDirac(core_orbitals.back(), en_a, log_dele_or);
    core_orbitals.back().occ_frac = double(num) / (4 * l + 2);
  }
}

//******************************************************************************
void Wavefunction::solveNewValence(int n, int k, double en_a, int log_dele_or)
// Update to take a list ok nken's ?
{
  valence_orbitals.emplace_back(DiracSpinor{n, k, rgrid});

  // Solve local dirac Eq:
  auto &psi = valence_orbitals.back();
  if (en_a == 0)
    en_a = enGuessVal(n, k);
  solveDirac(psi, en_a, log_dele_or);
}

//******************************************************************************
void Wavefunction::orthonormaliseOrbitals(std::vector<DiracSpinor> &in_orbs,
                                          int num_its)
// Note: this function is static
// Forces ALL orbitals to be orthogonal to each other, and normal
// Note: workes best if run twice!
// |a> ->  |a> - \sum_{b!=a} |b><b|a>
// Then:
// |a> -> |a> / <a|a>
// c_ba = c_ab = <a|b>
// num_its is optional parameter. Repeats that many times!
// Note: I force all orthog to each other - i.e. double count.
// {force <2|1>=0 and then <1|2>=0}
// Would be 2x faster not to do this
//  - but that would treat some orbitals special!
// Hence factor of 0.5
// Note: For HF, should never be called after core is frozen!
//
// Note: This allows wfs to extend past pinf!
// ==> This causes the possible orthog issues..
{
  auto Ns = in_orbs.size();

  // Calculate c_ab = <a|b>  [only for b>a -- symmetric]
  std::vector<std::vector<double>> c_ab(Ns, std::vector<double>(Ns));
  for (std::size_t a = 0; a < Ns; a++) {
    const auto &phi_a = in_orbs[a];
    for (auto b = a + 1; b < Ns; b++) {
      const auto &phi_b = in_orbs[b];
      if (phi_a.k != phi_b.k) //|| phi_a.n == phi_b.n - can't happen!
        continue;
      c_ab[a][b] = 0.5 * (phi_a * phi_b);
    }
  }
  // note: above loop executes psia*psib half as many times as below would

  // Orthogonalise + re-norm orbitals:
  for (std::size_t a = 0; a < Ns; a++) {
    auto &phi_a = in_orbs[a];
    for (std::size_t b = 0; b < Ns; b++) {
      const auto &phi_b = in_orbs[b];
      if (phi_a.k != phi_b.k || phi_a.n == phi_b.n)
        continue;
      double cab = (a < b) ? c_ab[a][b] : c_ab[b][a];
      phi_a -= cab * phi_b;
    }
    phi_a.normalise();
  }

  // If necisary: repeat
  if (num_its > 1)
    orthonormaliseOrbitals(in_orbs, num_its - 1);
}

//******************************************************************************
void Wavefunction::orthogonaliseWrt(DiracSpinor &psi_v,
                                    const std::vector<DiracSpinor> &in_orbs) {
  for (const auto &psi_c : in_orbs) {
    if (psi_v.k != psi_c.k)
      continue;
    psi_v -= (psi_v * psi_c) * psi_c;
  }
}
//******************************************************************************
void Wavefunction::orthonormaliseWrt(DiracSpinor &psi_v,
                                     const std::vector<DiracSpinor> &in_orbs)
// Static.
// Force given orbital to be orthogonal to all core orbitals
// [After the core is 'frozen', don't touch core orbitals!]
// |v> --> |v> - sum_c |c><c|v>
// note: here, c denotes core orbitals
{
  orthogonaliseWrt(psi_v, in_orbs);
  psi_v.normalise();
}

//******************************************************************************
std::tuple<double, double> Wavefunction::lminmax_core_range(int l,
                                                            double eps) const {

  std::vector<double> rho_l(rgrid.num_points);
  for (const auto &Fc : core_orbitals) {
    if (Fc.l() == l)
      rho_l = NumCalc::add_vectors(rho_l, Fc.rho());
  }
  // find maximum rho:
  auto max = std::max_element(rho_l.begin(), rho_l.end());
  // find first position that rho=|psi^2| reaches cut-off
  auto cut = eps * (*max);
  auto lam = [=](const auto &v) { return cut < v; };
  auto first = std::find_if(rho_l.begin(), rho_l.end(), lam);
  auto last = std::find_if(rho_l.rbegin(), rho_l.rend(), lam);
  auto index_first = std::size_t(first - rho_l.begin());
  auto index_last = std::size_t(rho_l.rend() - last);
  // std::cout << l << " " << index_first << " " << rgrid.r[index_first] << " -
  // "<< index_last << " " << rgrid.r[index_last] << "\n";
  return {rgrid.r[index_first], rgrid.r[index_last]};
}

//******************************************************************************
double Wavefunction::enGuessCore(int n, int l) const
// Private
// Energy guess for core states. Not perfect, good enough
// num_el_below = total electrons BELOW
// num = num electrons in THIS shell
{
  int num_el_below = 0;
  int num_el_this = 0;
  for (const auto &config : m_core_configs) {
    if (l == config.l && n == config.n) {
      num_el_this = config.num;
      break;
    }
    num_el_below += config.num;
  }

  // effective Z (for energy guess) -- not perfect!
  double Zeff = 1.0 + (m_Z - num_el_below - 0.5 * num_el_this);
  if (Zeff < 1.0) {
    Zeff = 1.;
  }

  double en_a = -0.5 * std::pow(Zeff / n, 2);
  if (n > 1) {
    en_a *= 0.5;
  }
  if (Zeff < 10) {
    if (l == 0)
      en_a *= 2.5;
    if (l == 1)
      en_a *= 3.5;
  }

  return en_a;
}

//******************************************************************************
double Wavefunction::enGuessVal(int n, int ka) const
// Energy guess for valence states. Not perfect, good enough
{
  int maxn = maxCore_n();
  int l = AtomData::l_k(ka);
  int dn = n - maxn;
  double neff = 1.0 + dn;
  double x = 1;
  auto Z_eff = m_Z - num_core_electrons;
  if (maxn < 4)
    x = 0.25;
  if (l == 1)
    neff += 0.5 * x;
  if (l == 2)
    neff += 2.0 * std::pow(x, 0.5);
  if (l >= 3)
    neff += 4.0 * x;
  return -0.5 * Z_eff * Z_eff / std::pow(neff, 2);
}

//******************************************************************************
std::string Wavefunction::nuclearParams() const {
  std::ostringstream output;

  auto rrms = m_nuc_params.r_rms;
  auto t = m_nuc_params.t;

  switch (m_nuc_params.type) {
  case Nuclear::Type::point:
    output << "Point-like nucleus; ";
    break;
  case Nuclear::Type::spherical:
    output << "Spherical nucleus; "
           << " r_rms = " << rrms
           << ", r_charge = " << Nuclear::c_hdr_formula_rrms_t(rrms, 0);
    break;
  case Nuclear::Type::Fermi:
    output << "Fermi nucleus; "
           << " r_rms = " << rrms
           << ", c_hdr = " << Nuclear::c_hdr_formula_rrms_t(rrms, t)
           << ", t = " << t;
    break;
  }
  return output.str();
}

//******************************************************************************
std::vector<std::size_t>
Wavefunction::sortedEnergyList(const std::vector<DiracSpinor> &tmp_orbs,
                               bool do_sort)
// Static
// Outouts a list of integers corresponding to the states
// sorted by energy (lowest energy first)
{

  using DoubleInt = std::pair<double, std::size_t>;
  std::vector<DoubleInt> t_en;
  for (std::size_t i = 0; i < tmp_orbs.size(); i++) {
    t_en.emplace_back(tmp_orbs[i].en, i);
  }

  // Sort list of Pairs by first element in the pair:
  auto compareThePair = [](const DoubleInt &di1, const DoubleInt &di2) {
    return di1.first < di2.first;
  };

  if (do_sort)
    std::sort(t_en.begin(), t_en.end(), compareThePair);

  // overwrite list with sorted list
  std::vector<std::size_t> sorted_list;
  std::for_each(t_en.begin(), t_en.end(),
                [&](const DoubleInt &el) { sorted_list.push_back(el.second); });

  return sorted_list;
}

//******************************************************************************
void Wavefunction::printCore(bool sorted) const
// prints core orbitals
{
  int Zion = Znuc() - Ncore();
  std::cout << "Core: " << coreConfiguration_nice() << " (V^N";
  if (Zion != 0)
    std::cout << "-" << Zion;
  std::cout << ")\n";
  if (Ncore() < 1)
    return;

  std::cout
      << "     state   k   Rinf its   eps         En (au)        En (/cm)\n";
  auto index_list = sortedEnergyList(core_orbitals, sorted);
  for (auto i : index_list) {
    const auto &phi = core_orbitals[i];
    auto r_inf = rgrid.r[phi.pinf - 1]; // rinf(phi);
    printf("%2i) %7s %2i  %5.1f %2i  %5.0e %15.9f %15.3f", int(i),
           phi.symbol().c_str(), phi.k, r_inf, phi.its, phi.eps, phi.en,
           phi.en *PhysConst::Hartree_invcm);
    if (phi.occ_frac < 1.0)
      printf("     [%4.2f]\n", phi.occ_frac);
    else
      std::cout << "\n";
  }
  if (m_pHF) {
    auto core_energy = coreEnergyHF();
    printf("E_core = %.8g au; = %.8g /cm\n", core_energy,
           core_energy * PhysConst::Hartree_invcm);
  }
}

//******************************************************************************
void Wavefunction::printValence(
    bool sorted, const std::vector<DiracSpinor> &in_orbitals) const {
  auto tmp_orbs = (in_orbitals.empty()) ? valence_orbitals : in_orbitals;
  if (tmp_orbs.empty())
    return;

  // Find lowest valence energy:
  auto e0 = 0.0;
  for (auto &phi : tmp_orbs) {
    if (phi.en < e0)
      e0 = phi.en;
  }

  std::cout
      << "Val: state   "
      << "k   Rinf its   eps         En (au)        En (/cm)   En (/cm)\n";
  auto index_list = sortedEnergyList(tmp_orbs, sorted);
  for (auto i : index_list) {
    const auto &phi = tmp_orbs[i];
    auto r_inf = rgrid.r[phi.pinf - 1]; // rinf(phi);
    printf("%2i) %7s %2i  %5.1f %2i  %5.0e %15.9f %15.3f", int(i),
           phi.symbol().c_str(), phi.k, r_inf, phi.its, phi.eps, phi.en,
           phi.en *PhysConst::Hartree_invcm);
    printf(" %10.2f\n", (phi.en - e0) * PhysConst::Hartree_invcm);
  }
}

//******************************************************************************
void Wavefunction::printBasis(bool sorted) const {

  std::cout << "Basis: \n";
  std::cout
      << "     State   k  R0     Rinf          En(Basis)         En(HF)\n";
  auto index_list = sortedEnergyList(basis, sorted);
  for (auto i : index_list) {
    const auto &phi = basis[i];
    auto r_0 = rgrid.r[phi.p0];
    auto r_inf = rgrid.r[phi.pinf - 1];

    const auto *hf_phi = getState(phi.n, phi.k);
    if (hf_phi != nullptr) {
      // found HF state
      auto eps = 2.0 * (phi.en - hf_phi->en) / (phi.en + hf_phi->en);
      printf("%2i) %7s %2i  %5.0e %5.1f %18.7f  %13.7f  %6.0e\n", int(i),
             phi.symbol().c_str(), phi.k, r_0, r_inf, phi.en, hf_phi->en, eps);
    } else {
      printf("%2i) %7s %2i  %5.0e %5.1f %18.7f\n", int(i), phi.symbol().c_str(),
             phi.k, r_0, r_inf, phi.en);
    }
  }
}

//******************************************************************************
std::vector<AtomData::DiracSEnken>
Wavefunction::listOfStates_nk(int num_val, int la, int lb, bool skip_core) const
// Creates a list of states (usually valence states to solve for)
// In form {{n,ka},...}
// Outputs n and l, from min-max l (or from 0-> max l)
// Calulates num_val different n states (for each l)
// This will be number of states above the core (skips states which are in the
// core)
{
  std::vector<AtomData::DiracSEnken> lst;
  auto l_min = la;
  auto l_max = lb;
  if (lb == 0) {
    l_min = 0;
    l_max = la;
  }

  auto min_ik = AtomData::indexFromKappa(-l_min - 1);
  auto max_ik = AtomData::indexFromKappa(-l_max - 1);
  for (int ik = min_ik; ik <= max_ik; ik++) {
    auto k = AtomData::kappaFromIndex(ik);
    auto l = AtomData::l_k(k);
    auto n_min = l + 1;
    for (int n = n_min, count = 0; count < num_val; n++) {
      if (isInCore(n, k) && skip_core)
        continue;
      lst.emplace_back(n, k);
      ++count;
    }
  }
  return lst;
}

//******************************************************************************
std::vector<double> Wavefunction::coreDensity() const {
  std::vector<double> rho(rgrid.num_points, 0.0);
  for (const auto &phi : core_orbitals) {
    auto f = double(phi.twoj() + 1) * phi.occ_frac;
    for (auto i = 0ul; i < rgrid.num_points; i++) {
      rho[i] += f * (phi.f[i] * phi.f[i] + phi.g[i] * phi.g[i]);
    }
  }
  return rho;
}

//******************************************************************************
void Wavefunction::formBasis(const std::string &states_str,
                             const std::size_t n_spl, const std::size_t k_spl,
                             const double r0_spl, const double r0_eps,
                             const double rmax_spl, const bool positronQ) {
  basis = SplineBasis::form_basis(states_str, n_spl, k_spl, r0_spl, r0_eps,
                                  rmax_spl, *this, positronQ);
}

//******************************************************************************
std::vector<double> Wavefunction::get_Vlocal(int l) const {
  return NumCalc::add_vectors(vnuc, vdir, vrad.get_Vel(l));
}
//******************************************************************************
const std::vector<double> &Wavefunction::get_Hmag(int l) const {
  return vrad.get_Hmag(l);
}
