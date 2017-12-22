#ifndef DISTRIBUTED_BSPLINE_SET_H
#define DISTRIBUTED_BSPLINE_SET_H

#include "QMCWaveFunctions/EinsplineAdoptor.h"
#include <unistd.h>

namespace qmcplusplus
{

  template<typename PointType>
  struct BsplineExchangeData
  {
    enum class EvalQuantities
    { NONE, V, VGL, VGH };

    PointType position;
    EvalQuantities quantities;

    BsplineExchangeData() = default;
    BsplineExchangeData(const PointType& p, EvalQuantities q) :
      position(p), quantities(q)
    {
    }
  };

/** DistributedBsplineSet<SplineAdoptor>, a SPOSetBase
 * @tparam SplineAdoptor implements evaluation functions that matched
 * the storage requirements. 
 *
 * Equivalent to EinsplineSetExtended<Storage>
 * Storage is now handled by SplineAdoptor class that is specialized
 * for precision, storage etc. 
 * @todo Make SplineAdoptor be a member not the base class. This is needed
 * to make MultiBsplineSet (TBD) which has multiple SplineAdoptors for
 * distributed cases.
 */
template<typename SplineAdoptor>
class DistributedBsplineSet: public SPOSetBase, public SplineAdoptor
{
  typedef typename SplineAdoptor::SplineType SplineType;
  typedef typename SplineAdoptor::PointType  PointType;
  typedef ValueMatrix_t::value_type value_type;
  typedef GradMatrix_t::value_type grad_type;
  typedef HessMatrix_t::value_type hess_type;

  ValueMatrix_t send_values;
  std::vector<ValueMatrix_t> recv_values;
  std::vector<Communicate::request> send_requests;
  std::vector<Communicate::request> recv_requests;
  const int exchange_tag = 27183;
  using ExchangeType = BsplineExchangeData<PointType>;

  // 1 value + 3 gradient + 9 hessian
  static const int max_values_per_orbital = 13;

  
  inline VectorViewer<value_type>
  sendValues(int rank) 
  {
    return VectorViewer<value_type>(
      send_values[rank*max_values_per_orbital+0],local_spos);
  }

  inline VectorViewer<grad_type>
  sendGradients(int rank) 
  {
    return VectorViewer<grad_type>(
      (grad_type*)send_values[rank*max_values_per_orbital+1],local_spos);
  }

  inline VectorViewer<value_type>
  sendLaplacians(int rank) 
  {
    return VectorViewer<value_type>(
      send_values[rank*max_values_per_orbital+4],local_spos);
  }

  inline VectorViewer<hess_type>
  sendHessians(int rank) 
  {
    return VectorViewer<hess_type>(
      (hess_type*)send_values[rank*max_values_per_orbital+4],local_spos);
  }

  inline VectorViewer<value_type>
  sendVGL(int rank) 
  {
    return VectorViewer<value_type>(
      send_values[rank*max_values_per_orbital+0],5*local_spos);
  }

  inline VectorViewer<value_type>
  sendVGH(int rank) 
  {
    return VectorViewer<value_type>(
      send_values[rank*max_values_per_orbital+0],13*local_spos);
  }


  inline VectorViewer<value_type>
  recvValues(int rank) 
  {
    size_t size = (SplineAdoptor::dist_spo_offsets[rank+1] - 
                   SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<value_type>(recv_values[rank][0], size);
  }

  inline VectorViewer<grad_type>
  recvGradients(int rank) 
  {
    size_t size = (SplineAdoptor::dist_spo_offsets[rank+1] - 
                   SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<grad_type>((grad_type*)recv_values[rank][1], size);
  }

  inline VectorViewer<value_type>
  recvLaplacians(int rank) 
  {
    size_t size = (SplineAdoptor::dist_spo_offsets[rank+1] - 
                   SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<value_type>((value_type*)recv_values[rank][4], size);
  }

  inline VectorViewer<hess_type>
  recvHessians(int rank) 
  {
    size_t size = (SplineAdoptor::dist_spo_offsets[rank+1] - 
                   SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<hess_type>((hess_type*)recv_values[rank][4], size);
  }

  inline VectorViewer<value_type>
  recvVGL(int rank)
  {
    size_t size = 5*(SplineAdoptor::dist_spo_offsets[rank+1] - 
                     SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<value_type>(recv_values[rank][0], size);
  }

  inline VectorViewer<value_type>
  recvVGH(int rank)
  {
    size_t size = 13*(SplineAdoptor::dist_spo_offsets[rank+1] - 
                     SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<value_type>(recv_values[rank][0], size);
  }


  std::vector<ExchangeType> exchange_data;

  int group_size=0;
  int group_rank=0;
  int local_spos = 0;

  inline const typename SplineAdoptor::PointType
  groupPosition(int rank) 
  {
    return exchange_data[rank].position;
  }

  void
  exchangeData(const ExchangeType& my_data)
  {
    if (!group_size) {
      group_size = SplineAdoptor::dist_group_comm->size();
      group_rank = SplineAdoptor::dist_group_comm->rank();
    }
    SplineAdoptor::dist_group_comm->allgather(
      (char*)(&my_data), (char*)&(exchange_data[0]), 
      sizeof(ExchangeType));
    for (int i=0; i< group_size; ++i) {
      auto r = exchange_data[i].position;
    }
  }

public:


  ///** default constructor */
  //BsplineSet() { }

  SPOSetBase* makeClone() const
  {
    return new DistributedBsplineSet<SplineAdoptor>(*this);
  }

  /** set_spline to the big table
   * @param psi_r starting address of real part of psi(ispline)
   * @param psi_i starting address of imaginary part of psi(ispline)
   * @param twist twist id, reserved to sorted adoptor, ignored
   * @param ispline index of this spline function
   * @param level refinement level
   *
   * Each adoptor handles the map with respect to the twist, state index and refinement level
   */
  template<typename CT>
  void set_spline(CT* spline_r, CT* spline_i, int twist, int ispline, int level)
  {
    SplineAdoptor::set_spline(spline_r,spline_i,twist,ispline,level);
  }

  inline void
  waitOnReceives()
  {
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        MPI_Status recv_wait_status;
        if (recv_requests[rank]) {
          MPI_Wait(&(recv_requests[rank]), &recv_wait_status);
          recv_requests[rank] = nullptr;
        }
      }
    }
  }

  inline void
  waitOnSends()
  {
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        MPI_Status send_wait_status;
        if (send_requests[rank]) {
          MPI_Wait(&(send_requests[rank]), &send_wait_status);
          send_requests[rank] = nullptr;
        }
      }
    }
  }
 
  inline int
  evaluateForRemote()
  {
    int num_remote_evals = 0;
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        typename ExchangeType::EvalQuantities quant = 
          exchange_data[rank].quantities;
        PointType pos = exchange_data[rank].position;
        switch (quant) {
        case ExchangeType::EvalQuantities::NONE:
          break;
        case ExchangeType::EvalQuantities::V:
          {
            VectorViewer<value_type> v = sendValues(rank);
            SplineAdoptor::evaluate_v(pos, v);
            send_requests[rank] = 
              SplineAdoptor::dist_group_comm->isend(rank, exchange_tag, v);
          }
          num_remote_evals++;
          break;
        case ExchangeType::EvalQuantities::VGL:
          {
            VectorViewer<value_type> v   = sendValues(rank);
            VectorViewer<grad_type>  g   = sendGradients(rank);
            VectorViewer<value_type> l   = sendLaplacians(rank);
            VectorViewer<value_type> vgl = sendVGL(rank);
            SplineAdoptor::evaluate_vgl(groupPosition(rank), v, g, l);
            send_requests[rank] = 
              SplineAdoptor::dist_group_comm->isend(rank, exchange_tag, vgl);
          }
          num_remote_evals++;
          break;
        case ExchangeType::EvalQuantities::VGH:
          {
            VectorViewer<value_type> v   = sendValues(rank);
            VectorViewer<grad_type>  g   = sendGradients(rank);
            VectorViewer<hess_type>  h   = sendHessians(rank);
            VectorViewer<value_type> vgh = sendVGH(rank);
            SplineAdoptor::evaluate_vgh(groupPosition(rank), v, g, h);
            send_requests[rank] = 
              SplineAdoptor::dist_group_comm->isend(rank, exchange_tag, vgh);
          }
          num_remote_evals++;
          break;
        default:
          fprintf(stderr, "Logic error, unknown quantity code in "
                  "DistributedBsplineSet::evaluateForRemote\n");
        }
      }
    }
    return num_remote_evals;
  }


  inline void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi)
  {
    exchangeData(ExchangeType(P.R[iat], ExchangeType::EvalQuantities::V));

    // Post receives first
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        size_t offset = SplineAdoptor::dist_spo_offsets[rank];
        size_t size   = SplineAdoptor::dist_spo_offsets[rank+1] - offset;
        VectorViewer<value_type> v(&(psi[0])+offset,size);
        recv_requests[rank] = 
          SplineAdoptor::dist_group_comm->irecv(rank, exchange_tag, v);
      }
    }

    int num_evals = evaluateForRemote();

    // evaluate local part
    size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    VectorViewer<value_type> v(&(psi[0])+offset,local_spos);
    SplineAdoptor::evaluate_v(groupPosition(group_rank), v);


    // for (int rank=0; rank < group_size; ++rank) {
    //   if (rank != group_rank) {
    //     VectorViewer<value_type> v = sendValues(rank);
    //     SplineAdoptor::evaluate_v(groupPosition(rank), v);
    //     send_requests[rank] = 
    //       SplineAdoptor::dist_group_comm->isend(rank, exchange_tag, v);
    //   }
    //   else {
    //     size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    //     size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    //     VectorViewer<value_type> v(&(psi[0])+offset,local_spos);
    //     SplineAdoptor::evaluate_v(groupPosition(rank), v);
    //   }
    // }
    waitOnReceives();
    waitOnSends();

    //    SplineAdoptor::evaluate_v(P.R[iat],psi);
  }

  inline void evaluateValues(const ParticleSet& P, ValueMatrix_t& psiM)
  {
    ValueVector_t psi(psiM.cols());
    for(int iat=0; iat<P.getTotalNum(); ++iat)
    {
      evaluate(P, iat, psi);
      // SplineAdoptor::evaluate_v(P.R[iat],psi);
      std::copy(psi.begin(),psi.end(),psiM[iat]);
    }
  }

  inline void evaluate(const ParticleSet& P, int iat,
                       ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi)
  {
    exchangeData(ExchangeType(P.R[iat], ExchangeType::EvalQuantities::VGL));

    // Post receives first
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        VectorViewer<value_type> vgl = recvVGL(rank);
        recv_requests[rank] = 
          SplineAdoptor::dist_group_comm->irecv(rank, exchange_tag, vgl);
      }
    }
    
    evaluateForRemote();
    size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    VectorViewer<value_type> v(&(  psi[offset]),local_spos);
    VectorViewer<grad_type>  g(&( dpsi[offset]),local_spos);
    VectorViewer<value_type> l(&(d2psi[offset]),local_spos);
    SplineAdoptor::evaluate_vgl(groupPosition(group_rank), v, g, l);

    // for (int rank=0; rank < group_size; ++rank) {
    //   if (rank != group_rank) {
    //     VectorViewer<value_type> v   = sendValues(rank);
    //     VectorViewer<grad_type>  g   = sendGradients(rank);
    //     VectorViewer<value_type> l   = sendLaplacians(rank);
    //     VectorViewer<value_type> vgl = sendVGL(rank);
    //     SplineAdoptor::evaluate_vgl(groupPosition(rank), v, g, l);
    //     send_requests[rank] = 
    //       SplineAdoptor::dist_group_comm->isend(rank, exchange_tag, vgl);
    //   }
    //   else {
    //     size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    //     size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    //     VectorViewer<value_type> v(&(  psi[offset]),local_spos);
    //     VectorViewer<grad_type>  g(&( dpsi[offset]),local_spos);
    //     VectorViewer<value_type> l(&(d2psi[offset]),local_spos);
    //     SplineAdoptor::evaluate_vgl(groupPosition(rank), v, g, l);
    //   }
    // }
    // Wait for all receives to complete
    waitOnReceives();

    // Now copy received data to psi, dpsi, and d2psi
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        size_t offset = SplineAdoptor::dist_spo_offsets[rank];
        size_t size   = SplineAdoptor::dist_spo_offsets[rank+1] - offset;
        VectorViewer<value_type> v = recvValues(rank);
        VectorViewer<grad_type>  g = recvGradients(rank);
        VectorViewer<value_type> l = recvLaplacians(rank);
        simd::copy(&(  psi[offset]), &(v[0]), size);
        simd::copy(&( dpsi[offset]), &(g[0]), size);
        simd::copy(&(d2psi[offset]), &(l[0]), size);
      }
    }    
    
    waitOnSends();
    //    SplineAdoptor::evaluate_vgl(P.R[iat],psi,dpsi,d2psi);
  }

  inline void evaluate(const ParticleSet& P, int iat,
                       ValueVector_t& psi, GradVector_t& dpsi, HessVector_t& grad_grad_psi)
  {
    exchangeData(ExchangeType(P.R[iat], ExchangeType::EvalQuantities::VGH));

    // Post receives first
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        VectorViewer<value_type> vgh = recvVGH(rank);
        recv_requests[rank] = 
          SplineAdoptor::dist_group_comm->irecv(rank, exchange_tag, vgh);
      }
    }

    evaluateForRemote();
    // Local evaluation
    size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    VectorViewer<value_type> v(&(  psi[offset]),local_spos);
    VectorViewer<grad_type>  g(&( dpsi[offset]),local_spos);
    VectorViewer<hess_type>  h(&(grad_grad_psi[offset]),local_spos);
    SplineAdoptor::evaluate_vgh(groupPosition(group_rank), v, g, h);


    // for (int rank=0; rank < group_size; ++rank) {
    //   if (rank != group_rank) {
    //     VectorViewer<value_type> v   = sendValues(rank);
    //     VectorViewer<grad_type>  g   = sendGradients(rank);
    //     VectorViewer<hess_type> h   = sendHessians(rank);
    //     VectorViewer<value_type> vgh = sendVGH(rank);
    //     SplineAdoptor::evaluate_vgh(groupPosition(rank), v, g, h);
    //     send_requests[rank] = 
    //       SplineAdoptor::dist_group_comm->isend(rank, exchange_tag, vgh);
    //   }
    //   else {
    //     size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    //     size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    //     VectorViewer<value_type> v(&(  psi[offset]),local_spos);
    //     VectorViewer<grad_type>  g(&( dpsi[offset]),local_spos);
    //     VectorViewer<hess_type>  h(&(grad_grad_psi[offset]),local_spos);
    //     SplineAdoptor::evaluate_vgh(groupPosition(rank), v, g, h);
    //   }
    // }

    // Wait for all receives to complete
    waitOnReceives();

    // Now copy received data to psi, dpsi, and d2psi
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        size_t offset = SplineAdoptor::dist_spo_offsets[rank];
        size_t size   = SplineAdoptor::dist_spo_offsets[rank+1] - offset;
        VectorViewer<value_type> v = recvValues(rank);
        VectorViewer<grad_type>  g = recvGradients(rank);
        VectorViewer<hess_type>  h = recvHessians(rank);
        simd::copy(&(  psi[offset]), &(v[0]), size);
        simd::copy(&( dpsi[offset]), &(g[0]), size);
        simd::copy(&(grad_grad_psi[offset]), &(h[0]), size);
      }
    }    
    
    waitOnSends();
    //    SplineAdoptor::evaluate_vgh(P.R[iat],psi,dpsi,grad_grad_psi);
  }

  void resetParameters(const opt_variables_type& active)
  { }

  void resetTargetParticleSet(ParticleSet& e)
  { }

  void setOrbitalSetSize(int norbs)
  {
    OrbitalSetSize = norbs;
    BasisSetSize=norbs;

    //SplineAdoptor::first_spo=0;
    //SplineAdoptor::last_spo=norbs;
  }

  virtual void
  resizeRemoteStorage(size_t group_size, size_t num_local_spos) override
  {
    local_spos = num_local_spos;
    send_values.resize(group_size*max_values_per_orbital, num_local_spos);
    exchange_data.resize(group_size);
    recv_values.resize(group_size);
    send_requests.resize(group_size);
    recv_requests.resize(group_size);
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != SplineAdoptor::dist_group_comm->rank()) {
        size_t num_orbs = SplineAdoptor::dist_spo_offsets[rank+1] -
          SplineAdoptor::dist_spo_offsets[rank];
        recv_values[rank].resize(max_values_per_orbital, num_orbs);
      }
    }
  }



  void 
  evaluate_notranspose(const ParticleSet& P, int first, int last, 
                       ValueMatrix_t& logdet, GradMatrix_t& dlogdet, 
                       ValueMatrix_t& d2logdet)
  {
    typedef ValueMatrix_t::value_type value_type;
    typedef GradMatrix_t::value_type grad_type;

    ValueVector_t   psi(OrbitalSetSize);
    GradVector_t   dpsi(OrbitalSetSize);
    ValueVector_t d2psi(OrbitalSetSize);

    for(int iat=first, i=0; iat<last; ++iat,++i)
    {
      VectorViewer<value_type> v(logdet[i],OrbitalSetSize);
      VectorViewer<grad_type>  g(dlogdet[i],OrbitalSetSize);
      VectorViewer<value_type> l(d2logdet[i],OrbitalSetSize);
      evaluate(P, iat, psi, dpsi, d2psi);
      simd::copy(&(v[0]), &(  psi[0]), OrbitalSetSize);
      simd::copy(&(g[0]), &( dpsi[0]), OrbitalSetSize);
      simd::copy(&(l[0]), &(d2psi[0]), OrbitalSetSize);
      //      SplineAdoptor::evaluate_vgl(P.R[iat],v,g,l);
    }
  }

  virtual void 
  evaluate_notranspose(const ParticleSet& P, int first, int last, 
                       ValueMatrix_t& logdet, GradMatrix_t& dlogdet, 
                       HessMatrix_t& grad_grad_logdet)
  {
    typedef ValueMatrix_t::value_type value_type;
    typedef GradMatrix_t::value_type grad_type;
    typedef HessMatrix_t::value_type hess_type;

    ValueVector_t  psi(OrbitalSetSize);
    GradVector_t  dpsi(OrbitalSetSize);
    HessVector_t d2psi(OrbitalSetSize);

    for(int iat=first, i=0; iat<last; ++iat,++i)
    {
      VectorViewer<value_type> v(  logdet[i],OrbitalSetSize);
      VectorViewer<grad_type>  g( dlogdet[i],OrbitalSetSize);
      VectorViewer<hess_type>  h(grad_grad_logdet[i],OrbitalSetSize);
      evaluate(P, iat, psi, dpsi, d2psi);
      simd::copy(&(v[0]), &(  psi[0]), OrbitalSetSize);
      simd::copy(&(g[0]), &( dpsi[0]), OrbitalSetSize);
      simd::copy(&(h[0]), &(d2psi[0]), OrbitalSetSize);
      //      SplineAdoptor::evaluate_vgh(P.R[iat],v,g,h);
    }
  }

  /** einspline does not need any other state data */
  void evaluateVGL(const ParticleSet& P, int iat, VGLVector_t& vgl, bool newp)
  {
    // app_log() << "evaluateVGL" << std::endl;
    abort();
    SplineAdoptor::evaluate_vgl_combo(P.R[iat],vgl);
  }

  virtual void
  completeDistributedEvaluations(int generation) override
  {
    ExchangeType exch(PointType(),ExchangeType::EvalQuantities::NONE);
    do {
      exchangeData(exch);
    } while (evaluateForRemote());
  }

  virtual Communicate*
  getDistributedOrbitalComm() const
  {
    return SplineAdoptor::dist_group_comm;
  }



};

}

#endif
