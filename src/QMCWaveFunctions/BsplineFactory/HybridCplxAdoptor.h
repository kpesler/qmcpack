//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
// File created by: Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//
//////////////////////////////////////////////////////////////////////////////////////


/** @file HybridCplxAdoptor.h
 *
 * Adoptor classes to handle complex hybrid orbitals with arbitrary precision
 */
#ifndef QMCPLUSPLUS_HYBRID_CPLX_SOA_ADOPTOR_H
#define QMCPLUSPLUS_HYBRID_CPLX_SOA_ADOPTOR_H

#include <QMCWaveFunctions/BsplineFactory/HybridAdoptorBase.h>
namespace qmcplusplus
{

/** adoptor class to match
 *
 */
template<typename BaseAdoptor>
struct HybridCplxSoA: public BaseAdoptor, public HybridAdoptorBase<typename BaseAdoptor::DataType>
{
  using HybridBase       = HybridAdoptorBase<typename BaseAdoptor::DataType>;
  using ST               = typename BaseAdoptor::DataType;
  using PointType        = typename BaseAdoptor::PointType;
  using SingleSplineType = typename BaseAdoptor::SingleSplineType;

  using BaseAdoptor::myV;
  using BaseAdoptor::myG;
  using BaseAdoptor::myL;
  using BaseAdoptor::myH;

  HybridCplxSoA(): BaseAdoptor()
  {
    this->AdoptorName="Hybrid"+this->AdoptorName;
    this->KeyWord="Hybrid"+this->KeyWord;
  }

  inline void resizeStorage(size_t n, size_t nvals)
  {
    BaseAdoptor::resizeStorage(n,nvals);
    HybridBase::resizeStorage(myV.size());
  }

  void bcast_tables(Communicate* comm)
  {
    BaseAdoptor::bcast_tables(comm);
    HybridBase::bcast_tables(comm);
  }

  void reduce_tables(Communicate* comm)
  {
    BaseAdoptor::reduce_tables(comm);
    HybridBase::reduce_atomic_tables(comm);
  }

  bool read_splines(hdf_archive& h5f)
  {
    return BaseAdoptor::read_splines(h5f) && HybridBase::read_splines(h5f);
  }

  bool write_splines(hdf_archive& h5f)
  {
    return BaseAdoptor::write_splines(h5f) && HybridBase::write_splines(h5f);
  }

  inline void flush_zero()
  {
    BaseAdoptor::flush_zero();
    HybridBase::flush_zero();
  }

  template<typename VV>
  inline void evaluate_v(const ParticleSet& P, const int iat, VV& psi)
  {
    if(HybridBase::evaluate_v(P,iat,myV))
    {
      const PointType& r=P.R[iat];
      BaseAdoptor::assign_v(r,psi);
    }
    else
      BaseAdoptor::evaluate_v(P,iat,psi);
  }

  template<typename TT>
  inline TT evaluate_dot(const ParticleSet& P, int iat, const TT* restrict arow, ST* scratch)
  {
    Vector<ST> vtmp(scratch,myV.size());
    if(HybridBase::evaluate_v(P,iat,vtmp))
      return BaseAdoptor::evaluate_dot(P,iat,arow,scratch,false);
    else
      return BaseAdoptor::evaluate_dot(P,iat,arow,scratch,true);
  }

  template<typename VV, typename GV>
  inline void evaluate_vgl(const ParticleSet& P, const int iat, VV& psi, GV& dpsi, VV& d2psi)
  {
    if(HybridBase::evaluate_vgl(P,iat,myV,myG,myL))
    {
      const PointType& r=P.R[iat];
      BaseAdoptor::assign_vgl_from_l(r,psi,dpsi,d2psi);
    }
    else
      BaseAdoptor::evaluate_vgl(P,iat,psi,dpsi,d2psi);
  }

  /** evaluate VGL using VectorSoaContainer
   * @param r position
   * @param psi value container
   * @param dpsi gradient-laplacian container
   */
  template<typename VGL>
  inline void evaluate_vgl_combo(const ParticleSet& P, const int iat, VGL& vgl)
  {
    if(HybridBase::evaluate_vgh(P,iat,myV,myG,myH))
    {
      const PointType& r=P.R[iat];
      BaseAdoptor::assign_vgl_soa(r,vgl);
    }
    else
      BaseAdoptor::evaluate_vgl_combo(P,iat,vgl);
  }

  template<typename VV, typename GV, typename GGV>
  inline void evaluate_vgh(const ParticleSet& P, const int iat, VV& psi, GV& dpsi, GGV& grad_grad_psi)
  {
    if(HybridBase::evaluate_vgh(P,iat,myV,myG,myH))
    {
      const PointType& r=P.R[iat];
      BaseAdoptor::assign_vgh(r,psi,dpsi,grad_grad_psi);
    }
    else
      BaseAdoptor::evaluate_vgh(P,iat,psi,dpsi,grad_grad_psi);
  }
};

}
#endif
