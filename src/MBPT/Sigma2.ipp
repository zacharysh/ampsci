#pragma once
#include "Angular/include.hpp"
#include "Coulomb/include.hpp"
#include "Wavefunction/DiracSpinor.hpp"

namespace MBPT {

//! Matrix element of 1-body Sigma (2nd-order correlation) operator;
//! de_v = <v|Sigma|v>. qk (CoulombIntegral) may be YkTable or QkTable.
template <class CoulombIntegral> // CoulombIntegral may be YkTable or QkTable
double Sigma_vw(const DiracSpinor &v, const DiracSpinor &w,
                const CoulombIntegral &qk, const std::vector<DiracSpinor> &core,
                const std::vector<DiracSpinor> &excited, int max_l_internal,
                std::optional<double> ev) {

  static_assert(std::is_same_v<CoulombIntegral, Coulomb::QkTable> ||
                std::is_same_v<CoulombIntegral, Coulomb::YkTable>);

  // const auto e_v = ev ? *ev : v.en();

  const auto e_v = ev ? *ev : 0.5 * (v.en() + w.en());
  // Should be e_v, OR e_w, depending on time ordering. We use average to make
  // diagrams symmetric.

  // Calculates <Fv|Sigma|Fw> from scratch, at Fw energy [full grid + fg+gg]
  if (v.kappa() != w.kappa())
    return 0.0;

  // Mainly for tests, but can include basis states only up to maximum l
  if (max_l_internal < 0)
    max_l_internal = 999;

  double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) collapse(2)
  for (auto ia = 0ul; ia < core.size(); ia++) {
    for (auto in = 0ul; in < excited.size(); in++) {

      const auto &a = core[ia];
      const auto &n = excited[in];

      if (a.l() > max_l_internal || n.l() > max_l_internal)
        continue;

      const auto de_an = a.en() - n.en();

      // Diagrams (a) [direct] and (b) [exchange]
      for (const auto &m : excited) {
        if (m.l() > max_l_internal)
          continue;

        const auto inv_de = 1.0 / (e_v + de_an - m.en());

        const auto [k0, k1] = Coulomb::k_minmax_Q(v, a, m, n);
        for (int k = k0; k <= k1; k += 2) {
          const auto Qk_vamn = qk.Q(k, v, a, m, n);
          const auto Qk_wamn = (v == w) ? Qk_vamn : qk.Q(k, w, a, m, n);
          const auto Pk_wamn = qk.P(k, w, a, m, n);
          sum += Qk_vamn * (Qk_wamn + Pk_wamn) * inv_de / (2.0 * k + 1.0);
        }
      }

      // Diagrams (c) [direct] and (d) [exchange]
      for (const auto &b : core) {
        if (b.l() > max_l_internal)
          continue;

        const auto inv_de = 1.0 / (e_v - de_an - b.en()); // v? w? ... careful!

        const auto [k0, k1] = Coulomb::k_minmax_Q(v, n, b, a);
        for (int k = k0; k <= k1; k += 2) {
          const auto Qk_vnba = qk.Q(k, v, n, b, a);
          const auto Qk_wnba = (v == w) ? Qk_vnba : qk.Q(k, w, n, b, a);
          const auto Pk_wnba = qk.P(k, w, n, b, a);
          sum += Qk_vnba * (Qk_wnba + Pk_wnba) * inv_de / (2.0 * k + 1.0);
        }
      }

    } // n
  }   // a

  return sum * (1.0 / v.twojp1());
}

} // namespace MBPT