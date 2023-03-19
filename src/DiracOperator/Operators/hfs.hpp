#pragma once
#include "DiracOperator/TensorOperator.hpp"
#include "IO/InputBlock.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "fmt/color.hpp"
#include "qip/Maths.hpp"
#include <functional>

namespace DiracOperator {

//==============================================================================

//! Functions for F(r) [nuclear magnetisation distribution] and similar
namespace Hyperfine {
using Func_R2_R = std::function<double(double, double)>; // save typing

//! Spherical ball F(r): (r/rN)^3 for r<rN, 1 for r>rN
inline auto sphericalBall_F() -> Func_R2_R {
  return [=](double r, double rN) {
    return (r > rN) ? 1.0 : (r * r * r) / (rN * rN * rN);
  };
}

//! Spherical shell F(r): 0 for r<rN, 1 for r>rN
inline auto sphericalShell_F() -> Func_R2_R {
  return [=](double r, double rN) { return (r > rN) ? 1.0 : 0.0; };
}

//! Pointlike F(r): 1
inline auto pointlike_F() -> Func_R2_R {
  return [=](double, double) { return 1.0; };
}

//------------------------------------------------------------------------------
//! 'Volotka' single-particle model: see Phys. Rev. Lett. 125, 063002 (2020)
inline auto VolotkaSP_F(double mu, double I_nuc, double l_pn, int gl)
    -> Func_R2_R
// Function that returns generates + returns F_BW Bohr-Weiskopf
// gl = 1 for proton, =0 for neutron. Double allowed for testing..
// mu is in units of Nuclear Magneton!
{
  const auto two_I = int(2 * I_nuc + 0.0001);
  const auto two_l = int(2 * l_pn + 0.0001);
  const auto g_l = double(gl); // just safety
  const auto gI = mu / I_nuc;

  const auto K = (l_pn * (l_pn + 1.0) - (3. / 4.)) / (I_nuc * (I_nuc + 1.0));
  const double g_s = (2.0 * gI - g_l * (K + 1.0)) / (1.0 - K);
  std::cout << "BW using: gl=" << g_l << ", gs=" << g_s << ", l=" << l_pn
            << ", gI=" << gI << " (I=" << I_nuc << ")\n";
  const double factor =
      (two_I == two_l + 1) ?
          g_s * (1 - two_I) / (4.0 * (two_I + 2)) + g_l * 0.5 * (two_I - 1) :
          g_s * (3 + two_I) / (4.0 * (two_I + 2)) +
              g_l * 0.5 * two_I * (two_I + 3) / (two_I + 2);
  if (two_I != two_l + 1 && two_I != two_l - 1) {
    std::cerr << "\nFAIL:59 in Hyperfine (VolotkaSP_F):\n "
                 "we must have I = l +/- 1/2, but we have: I,l = "
              << I_nuc << "," << l_pn << "\n";
    return [](double, double) { return 0.0; };
  }
  return [=](double r, double rN) {
    return (r > rN) ? 1.0 :
                      ((r * r * r) / (rN * rN * rN)) *
                          (1.0 - (3.0 / mu) * std::log(r / rN) * factor);
  };
}

//------------------------------------------------------------------------------
//! 'Volotka' SP model, for doubly-odd nuclei: Phys. Rev. Lett. 125, 063002 (2020)
inline auto doublyOddSP_F(double mut, double It, double mu1, double I1,
                          double l1, int gl1, double I2, double l2) -> Func_R2_R
// F(r) * g = 0.5 [ g1F1 + g2F2 + (g1F1 - g2F2) * K]
// K = [I1(I1+1) - I2(I2+1)] / [I(I+1)]
// return F(r) [divide by g]
// VolotkaSP_F(mu, I, l, g_l); //gl is 1 or 0
// g2 : from: g = 0.5 [ g1 + g2 + (g1 - g2) * K]
{
  const auto K = (I1 * (I1 + 1.0) - I2 * (I2 + 1.0)) / (It * (It + 1.0));
  const auto gt = mut / It;
  const auto g1 = mu1 / I1;
  const auto g2 = (g1 * (K + 1.0) - 2.0 * gt) / (K - 1.0);
  const auto mu2 = g2 * I2;
  const auto gl2 = (gl1 == 0) ? 1 : 0;
  const auto F1 = VolotkaSP_F(mu1, I1, l1, gl1);
  const auto F2 = VolotkaSP_F(mu2, I2, l2, gl2);
  return [=](double r, double rN) {
    return (0.5 / gt) * (g1 * F1(r, rN) + g2 * F2(r, rN) +
                         K * (g1 * F1(r, rN) - g2 * F2(r, rN)));
  };
}

//------------------------------------------------------------------------------
//! Converts reduced matrix element to A/B coeficients
inline double convert_RME_to_AB(int k, int ka, int kb) {
  if (k > 2)
    return 1.0;
  // nb: only makes sense if ka==kb?
  const auto magnetic = k % 2 != 0;
  const auto angular_F = magnetic ? (ka + kb) * Angular::Ck_kk(k, -ka, kb) :
                                    -Angular::Ck_kk(k, ka, kb);
  const auto j = 0.5 * Angular::twoj_k(ka);
  const auto cA = ka / (j * (j + 1.0));
  const auto cB = (2.0 * j - 1.0) / (2.0 * j + 2.0);
  return magnetic ? cA / angular_F : cB / angular_F;
}

inline double hfsA(const TensorOperator *h, const DiracSpinor &Fa) {
  auto Raa = h->radialIntegral(Fa, Fa); //?
  return Raa * Fa.kappa() / (Fa.jjp1()) * PhysConst::muN_CGS_MHz;
}
inline double hfsB(const TensorOperator *h, const DiracSpinor &Fa) {
  auto Raa = h->radialIntegral(Fa, Fa); //?
  return Raa * double(Fa.twoj() - 1) / double(Fa.twoj() + 2) *
         PhysConst::barn_MHz;
}

//==============================================================================
//! Takes in F(r) and k, and forms hyperfine radial function: F(r,rN)/r^{k+1}
inline std::vector<double> RadialFunc(int k, double rN, const Grid &rgrid,
                                      const Func_R2_R &hfs_F) {
  std::vector<double> rfunc;
  rfunc.reserve(rgrid.num_points());
  for (const auto r : rgrid)
    rfunc.push_back(hfs_F(r, rN) / pow(r, k + 1));
  return rfunc;
}

} // namespace Hyperfine

//==============================================================================
//! Units: Assumes g in nuc. magneton units (magnetic), and Q in barns
//! (electric)
class hfs final : public TensorOperator {
  // see Xiao, ..., Derevianko, Phys. Rev. A 102, 022810 (2020).
  using Func_R2_R = std::function<double(double, double)>;

public:
  hfs(int in_k, double in_GQ, double rN, const Grid &rgrid,
      const Func_R2_R &hfs_F = Hyperfine::pointlike_F(), bool MHzQ = true)
      : TensorOperator(in_k, Parity::even, in_GQ,
                       Hyperfine::RadialFunc(in_k, rN, rgrid, hfs_F), 0),
        k(in_k),
        magnetic(k % 2 != 0),
        cfg(magnetic ? 1.0 : 0.0),
        cff(magnetic ? 0.0 : 1.0),
        mMHzQ(MHzQ) {}

  std::string name() const override final {
    return "hfs" + std::to_string(k) + "";
  }
  std::string units() const override final {
    return (k <= 2 && mMHzQ) ? "MHz" : "au";
  }

  double angularF(const int ka, const int kb) const override final {
    // inludes unit: Assumes g in nuc. magneton units, and/or Q in barns
    // This only for k=1 (mag dipole) and k=2 E. quad.
    const auto unit = (mMHzQ && k == 1) ? PhysConst::muN_CGS_MHz :
                      (mMHzQ && k == 2) ? PhysConst::barn_MHz :
                                          1.0;
    return magnetic ? (ka + kb) * Angular::Ck_kk(k, -ka, kb) * unit :
                      -Angular::Ck_kk(k, ka, kb) * unit;
  }

  double angularCff(int, int) const override final { return cff; }
  double angularCgg(int, int) const override final { return cff; }
  double angularCfg(int, int) const override final { return cfg; }
  double angularCgf(int, int) const override final { return cfg; }

private:
  int k;
  bool magnetic;
  double cfg;
  double cff;
  bool mMHzQ;
};

//==============================================================================
//==============================================================================
inline std::unique_ptr<DiracOperator::TensorOperator>
generate_hfs(const IO::InputBlock &input, const Wavefunction &wf) {
  using namespace DiracOperator;

  input.check(
      {{"", "Most following will be taken from the default nucleus if "
            "not explicitely given"},
       {"mu", "Magnetic moment in mu_N"},
       {"Q", "Nuclear quadrupole moment, in barns. Also used as overall "
             "constant for any higher-order moments [1.0]"},
       {"k", "Multipolarity. 1=mag. dipole, 2=elec. quad, etc. [1]"},
       {"rrms",
        "nuclear (magnetic) rms radius, in Fermi (fm) (defult is charge rms)"},
       {"units", "Units for output (only for k=1,k=2). MHz or au [MHz]"},
       {"F(r)", "ball, point, shell, SingleParticle, or doublyOddSP [point]"},
       {"printF", "Writes F(r) to a text file [false]"},
       {"print", "Write F(r) info to screen [true]"},
       {"", "The following are only for SingleParticle or doublyOddSP"},
       {"I", "Nuclear spin. Taken from nucleus"},
       {"parity", "Nulcear parity: +/-1"},
       {"l", "l for unpaired nucleon (automatically derived from I and "
             "parity; best to leave as default)"},
       {"gl", "=1 for proton, =0 for neutron"},
       {"", "The following are only for doublyOddSP"},
       {"mu1", "mag moment of 'first' unpaired nucleon"},
       {"gl1", "gl of 'first' unpaired nucleon"},
       {"l1", "l of 'first' unpaired nucleon"},
       {"l2", "l of 'second' unpaired nucleon"},
       {"I1", "total spin (J) of 'first' unpaired nucleon"},
       {"I2", "total spin (J) of 'second' unpaired nucleon"}});
  if (input.has_option("help")) {
    return nullptr;
  }

  const auto nuc = wf.nucleus();
  const auto isotope = Nuclear::findIsotopeData(nuc.z(), nuc.a());
  const auto mu = input.get("mu", isotope.mu);
  const auto I_nuc = input.get("I", isotope.I_N);
  const auto print = input.get("print", true);
  const auto k = input.get("k", 1);

  const auto use_MHz =
      (k <= 2 &&
       qip::ci_compare(input.get<std::string>("units", "MHz"), "MHz"));

  if (k <= 0) {
    fmt::print(fg(fmt::color::red), "\nError 246:\n");
    std::cout << "In hyperfine: invalid K=" << k << "! meaningless results\n";
  }
  if (I_nuc <= 0) {
    fmt::print(fg(fmt::color::orange), "\nWarning 253:\n");
    std::cout << "In hyperfine: invalid I_nuc=" << I_nuc
              << "! meaningless results\n";
  }

  const auto g_or_Q = (k == 1) ? (mu / I_nuc) : input.get("Q", 1.0);

  enum class DistroType {
    point,
    ball,
    shell,
    SingleParticle,
    doublyOddSP,
    Error
  };

  const auto Fr_str = input.get<std::string>("F(r)", "point");
  const auto distro_type =
      (qip::ci_wc_compare(Fr_str, "point*") || qip::ci_compare(Fr_str, "1")) ?
          DistroType::point :
      qip::ci_compare(Fr_str, "ball")       ? DistroType::ball :
      qip::ci_compare(Fr_str, "shell")      ? DistroType::shell :
      qip::ci_wc_compare(Fr_str, "Single*") ? DistroType::SingleParticle :
      qip::ci_compare(Fr_str, "doublyOdd*") ? DistroType::doublyOddSP :
                                              DistroType::Error;
  if (distro_type == DistroType::Error) {
    fmt::print(fg(fmt::color::red), "\nError 271:\n");
    std::cout << "\nIn hyperfine. Unkown F(r) - " << Fr_str << "\n";
    std::cout << "Defaulting to pointlike!\n";
  }

  const auto r_rmsfm =
      distro_type == DistroType::point ? 0.0 : input.get("rrms", nuc.r_rms());
  const auto r_nucfm = std::sqrt(5.0 / 3) * r_rmsfm;
  const auto r_nucau = r_nucfm / PhysConst::aB_fm;

  if (print) {
    std::cout << "\nHyperfine structure: " << wf.atom() << "\n";
    std::cout << "K=" << k << " ("
              << (k == 1     ? "magnetic dipole" :
                  k == 2     ? "electric quadrupole" :
                  k % 2 == 0 ? "electric multipole" :
                               "magnetic multipole")
              << ")\n";
    std::cout << "Using " << Fr_str << " nuclear distro for F(r)\n"
              << "w/ r_N = " << r_nucfm << "fm = " << r_nucau
              << "au  (r_rms=" << r_rmsfm << "fm)\n";
    std::cout << "Points inside nucleus: " << wf.grid().getIndex(r_nucau)
              << "\n";
    if (k == 1) {
      std::cout << "mu = " << mu << ", I = " << I_nuc << ", g = " << g_or_Q
                << "\n";
    } else {
      std::cout << "Q = " << g_or_Q << "\n";
    }
  }

  // default is pointlike:
  auto Fr = Hyperfine::sphericalBall_F();
  if (distro_type == DistroType::ball) {
    Fr = Hyperfine::sphericalBall_F();
  } else if (distro_type == DistroType::shell) {
    Fr = Hyperfine::sphericalShell_F();
  } else if (distro_type == DistroType::SingleParticle) {
    const auto pi = input.get("parity", isotope.parity);
    const auto l_tmp = int(I_nuc + 0.5 + 0.0001);
    auto l = ((l_tmp % 2 == 0) == (pi == 1)) ? l_tmp : l_tmp - 1;
    l = input.get("l", l); // can override derived 'l' (not recommended)
    const auto gl_default = wf.Znuc() % 2 == 0 ? 0 : 1; // unparied proton?
    const auto gl = input.get<int>("gl", gl_default);
    if (print) {
      std::cout << "Single-Particle (Volotka formula) for unpaired";
      if (gl == 1)
        std::cout << " proton ";
      else if (gl == 0)
        std::cout << " neturon ";
      else
        std::cout << " gl=" << gl << "??? program will run, but prob wrong!\n";
      std::cout << "with l=" << l << " (pi=" << pi << ")\n";
    }
    Fr = Hyperfine::VolotkaSP_F(mu, I_nuc, l, gl);
  } else if (distro_type == DistroType::doublyOddSP) {
    const auto mu1 = input.get<double>("mu1", 1.0);
    const auto gl1 = input.get<int>("gl1", -1); // 1 or 0 (p or n)
    if (gl1 != 0 && gl1 != 1) {
      fmt::print(fg(fmt::color::red), "\nError 324:\n");
      std::cout << "In " << input.name() << " " << Fr_str
                << "; have gl1=" << gl1 << " but need 1 or 0\n";
      return std::make_unique<NullOperator>(NullOperator());
    }
    const auto l1 = input.get<double>("l1", -1.0);
    const auto l2 = input.get<double>("l2", -1.0);
    const auto I1 = input.get<double>("I1", -1.0);
    const auto I2 = input.get<double>("I2", -1.0);

    Fr = Hyperfine::doublyOddSP_F(mu, I_nuc, mu1, I1, l1, gl1, I2, l2);
  }

  // Optionally print F(r) function to file
  if (input.get<bool>("printF", false)) {
    std::ofstream of(wf.identity() + "_" + Fr_str + ".txt");
    of << "r/fm  F(r)\n";
    for (auto r : wf.grid()) {
      of << r * PhysConst::aB_fm << " "
         << Fr(r * PhysConst::aB_fm, r_nucau * PhysConst::aB_fm) << "\n";
    }
  }

  return std::make_unique<hfs>(k, g_or_Q, r_nucau, wf.grid(), Fr, use_MHz);
}

} // namespace DiracOperator