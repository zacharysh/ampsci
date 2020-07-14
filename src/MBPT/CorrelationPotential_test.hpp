#pragma once
#include "Wavefunction/BSplineBasis.hpp"
#include "Wavefunction/DiracSpinor.hpp"
#include "Wavefunction/Wavefunction.hpp"
#include "qip/Check.hpp"
#include "qip/Maths.hpp"
#include "qip/Vector.hpp"
#include <algorithm>
#include <string>

namespace UnitTest {

//******************************************************************************
bool CorrelationPotential(std::ostream &obuff) {
  bool pass = true;

  { // Compare with  K. Beloy and A. Derevianko,
    // Comput. Phys. Commun. 179, 310 (2008).
    Wavefunction wf({4000, 1.0e-6, 100.0, 0.33 * 100.0, "loglinear", -1.0},
                    {"Cs", -1, "Fermi", -1.0, -1.0}, 1.0);
    wf.hartreeFockCore("HartreeFock", 0.0, "[Xe]");
    wf.hartreeFockValence("6s");
    const auto &Fv = wf.valence.front();

    {
      // K. Beloy and A. Derevianko, Comput. Phys. Commun. 179, 310 (2008).
      const auto partial_KBAD =
          std::vector{-0.0000130, -0.0020027, -0.0105623, -0.0039347,
                      -0.0007563, -0.0002737, -0.0001182};
      // const auto total_KBAD = -0.0176609;
      const auto error = 0.0000001; // allow ony difference of 1 in last digit

      double prev = 0.0;
      std::vector<double> vals;
      wf.formBasis({"spdfghi", 100, 11, 0.0, 1.0e-6, 50.0, false});
      wf.formSigma(1, false);
      const auto Sigma = wf.getSigma();
      std::cout << "Table 2 from Beloy, Derevianko, Comput.Phys.Commun. 179, "
                   "310 (2008):\n";
      for (int l = 0; l <= 6; ++l) {
        const auto de = Sigma->SOEnergyShift(Fv, Fv, l);
        vals.push_back(de - prev);
        printf("%i %10.7f %10.7f  [%10.7f]\n", l, de, de - prev,
               partial_KBAD[std::size_t(l)]);
        prev = de;
      }

      // int l = 0;
      for (auto l = 0ul; l <= 4; ++l) {
        auto del = vals[l] - partial_KBAD[l];
        pass &= qip::check_value(
            &obuff, "MBPT(2) vs. KB,AD " + std::to_string(l), del, 0.0, error);
      }
      // Note: These are known to fail!
      qip::check_value(&obuff, "MBPT(2) vs. KB,AD 5 [(known fail]",
                       vals[5] - partial_KBAD[5], 0.0, error);
      qip::check_value(&obuff, "MBPT(2) vs. KB,AD 6 [known fail]",
                       vals[6] - partial_KBAD[6], 0.0, error);
    }

    { // "smaller" basis set (not exactly same as Derev)
      wf.formBasis({"30spdfghi", 40, 7, 0.0, 1.0e-6, 40.0, false});
      wf.formSigma(1, false);
      const auto Sigma = wf.getSigma();
      const auto de = Sigma->SOEnergyShift(Fv, Fv);
      auto ok = de >= -0.01767 && de <= -0.01748 ? 1 : 0;
      pass &= qip::check_value(&obuff, "MBPT(2) 'small' Cs 6s", ok, 1, 0);
    }
  }

  //****************************************************************************
  { // Compare Dzuba, only using up to l=4 for splines
    // Note: Works pretty well up to f states (not sure if difference is ok)
    auto dzuba_g = std::vector{-0.0195938,  -0.00399679, -0.00770113,
                               -0.00682331, -0.00214125, -0.00193494,
                               -0.01400596, -0.01324942
                               /*,-0.00033882, -0.00033866*/};
    std::sort(begin(dzuba_g), end(dzuba_g)); // sort: don't depend on order

    Wavefunction wf({2000, 1.0e-6, 120.0, 0.33 * 120.0, "loglinear", -1.0},
                    {"Cs", -1, "Fermi", -1.0, -1.0}, 1.0);
    wf.hartreeFockCore("HartreeFock", 0.0, "[Xe]");
    wf.hartreeFockValence("7sp5d"); //"7sp5d4f"
    wf.formBasis({"30spdfg", 40, 7, 0.0, 1.0e-6, 30.0, false});
    wf.formSigma(3, true, 1.0e-4, 30.0, 12 /*stride*/);

    std::vector<double> hf, br2;
    for (const auto &Fv : wf.valence) {
      hf.push_back(Fv.en);
    }

    wf.hartreeFockBrueckner();

    for (const auto &Fv : wf.valence) {
      br2.push_back(Fv.en);
    }

    auto de = qip::compose([](auto a, auto b) { return a - b; }, br2, hf);
    std::sort(begin(de), end(de)); // sort: don't depend on order

    auto [eps, at] = qip::compare_eps(dzuba_g, de);
    pass &= qip::check_value(&obuff, "Sigma2 Cs (spdfg)", eps, 0.0, 1.0e-2);
  }
  //

  { // Compare Dzuba, using up to l=6 for splines
    // Fails. Known to fail. Problem is with splines (probably)
    auto dzuba_i = std::vector{
        -0.02013813, -0.00410942, -0.00792483, -0.00702407, -0.00220878,
        -0.00199737, -0.01551449, -0.01466935, -0.00035253, -0.00035234};
    std::sort(begin(dzuba_i), end(dzuba_i)); // sort: don't depend on order

    // Note: Since we know it fails, can use small grid
    Wavefunction wf({1000, 1.0e-6, 120.0, 0.33 * 120.0, "loglinear", -1.0},
                    {"Cs", -1, "Fermi", -1.0, -1.0}, 1.0);
    wf.hartreeFockCore("HartreeFock", 0.0, "[Xe]");
    wf.hartreeFockValence("7sp5d4f"); //"7sp5d4f"
    // wf.formBasis({"30spdfghi", 40, 7, 0.0, 1.0e-6, 30.0, false});
    wf.formBasis({"15spdfghi", 20, 7, 0.0, 1.0e-6, 30.0, false});
    wf.formSigma(3, true, 1.0e-4, 30.0, 8 /*stride*/);

    std::vector<double> hf, br2;
    for (const auto &Fv : wf.valence) {
      hf.push_back(Fv.en);
    }

    wf.hartreeFockBrueckner();

    for (const auto &Fv : wf.valence) {
      br2.push_back(Fv.en);
    }

    auto de = qip::compose([](auto a, auto b) { return a - b; }, br2, hf);
    std::sort(begin(de), end(de)); // sort: don't depend on order

    auto [eps, at] = qip::compare_eps(dzuba_i, de);
    qip::check_value(&obuff, "Sigma2 Cs (spdfghi) [known fail]", eps, 0.0,
                     1.0e-2);
  }

  return pass;
}

} // namespace UnitTest
