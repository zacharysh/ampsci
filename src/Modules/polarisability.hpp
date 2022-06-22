#pragma once
#include "DiracOperator/DiracOperator.hpp"
#include "ExternalField/TDHF.hpp"
#include <vector>

// Forward declare classes:
class Wavefunction;
namespace IO {
class InputBlock;
}

namespace Module {

//! Calculate atomic (dipole) polarisability, alpha, at given frequency
/*! @details
 - J. Mitroy, M. S. Safronova, and C. W. Clark, Theory and Applications of
   Atomic and Ionic Polarizabilities, J. Phys. B 43, 44 (2010).
*/
void polarisability(const IO::InputBlock &input, const Wavefunction &wf);

//! Calculate dynamic (dipole) polarisability, alpha(w), as fn of frequency
void dynamicPolarisability(const IO::InputBlock &input, const Wavefunction &wf);

namespace alphaD {

//! Calculates polarisability of atomic core, using some-over-states method.
//! @details CorePolarisation (dVE1) is optional - assumed to be solved already.
double core_sos(const std::vector<DiracSpinor> &core,
                const std::vector<DiracSpinor> &spectrum,
                const DiracOperator::E1 &he1,
                const ExternalField::CorePolarisation *dVE1, double omega);

//! Calculates polarisability of valence state, using some-over-states method.
//! @details Total polarisabilitity is this + core. CorePolarisation (dVE1) is
//! optional - assumed to be solved already.
double valence_sos(const DiracSpinor &Fv,
                   const std::vector<DiracSpinor> &spectrum,
                   const DiracOperator::E1 &he1,
                   const ExternalField::CorePolarisation *dVE1, double omega);

//! Calculates polarisability of atomic core, using Mixed-States (TDHF) method.
/*! @details TDHF (dVE1) is required - assumed to be solved already. If it's not
solved, equivilant to no RPA.   See V. A. Dzuba, J. C. Berengut, J. S. M.
Ginges, and V. V. Flambaum, Screening of an Oscillating External Electric Field
in Atoms, Phys. Rev. A 98, 043411 (2018).
*/
double core_tdhf(const std::vector<DiracSpinor> &core,
                 const DiracOperator::E1 &he1, const ExternalField::TDHF &dVE1,
                 double omega, const MBPT::CorrelationPotential *const Sigma);

//! Calculates polarisability of va.ence state, using Mixed-States (TDHF) method
/*! @details TDHF (dVE1) is required - assumed to be solved already. If it's not
solved, equivilant to no RPA.   See V. A. Dzuba, J. C. Berengut, J. S. M.
Ginges, and V. V. Flambaum, Screening of an Oscillating External Electric Field
in Atoms, Phys. Rev. A 98, 043411 (2018).
*/
double valence_tdhf(const DiracSpinor &Fv, const DiracOperator::E1 &he1,
                    const ExternalField::TDHF &dVE1, double omega,
                    const MBPT::CorrelationPotential *const Sigma);

//! Calculates Structure-radiation + normalisation contribution to valence
//! polarisability, using sum-over-states.
/*! @details Spectrum is used for sum-over-states; hf_basis is used for MBPT.
 - delta_n_max_sum is largest \Delta n (from valence n) to include in sum over
states (~3 typical).
 - n_min_core is minimum core state n to include in MBPT Structure Radiation.
 - en_core is wf.en_coreval_gap() - used to separate core/excited.
 - Qk_fname is optional filename to read/write QkTable [speedup at memory cost].
 - See MBPT/StructureRad.hpp.
 */
double valence_SRN(const DiracSpinor &Fv,
                   const std::vector<DiracSpinor> &spectrum,
                   const DiracOperator::E1 &he1,
                   const ExternalField::CorePolarisation *dVE1, double omega,
                   // SR+N part:
                   int delta_n_max_sum, int n_min_core,
                   const std::vector<DiracSpinor> &hf_basis,
                   const double en_core, const std::string &Qk_fname = "");

} // namespace alphaD

} // namespace Module
