#ifndef DISTRIBUTED_BSPLINE_SET_H
#define DISTRIBUTED_BSPLINE_SET_H

#include "QMCWaveFunctions/EinsplineAdoptor.h"
#include <unistd.h>
#include <memory>
#include <omp.h>


//#define SEPARATE_WAITS

namespace qmcplusplus
{

  /** A simple data structure for storing both particle positions and
   *  the quantities that need to be cooperatively computed for a
   *  given walker.
   */
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
 *  @tparam SplineAdoptor implements evaluation functions that match
 *  the storage requirements. 
 *  This class implements a B-spline based SPOSet that utilizes MPI
 *  communication to allow the distribution of the large B-spline
 *  coefficient dataset between multiple nodes.  The dataset is split
 *  along the orbital index.  Tight coordination among members of each
 *  orbital group is required, as particle positions are exchanged and
 *  orbitals are evaluated by the node holding the coefficients for
 *  that orbital on behalf of remote nodes. 
 */
template<typename SplineAdoptor>
class DistributedBsplineSet: public SPOSetBase, public SplineAdoptor
{
  typedef typename SplineAdoptor::SplineType SplineType;
  typedef typename SplineAdoptor::PointType  PointType;
  typedef ValueMatrix_t::value_type value_type;
  typedef GradMatrix_t::value_type grad_type;
  typedef HessMatrix_t::value_type hess_type;

  std::vector<Communicate::request> requests;
  std::vector<Communicate::status>  statuses;
  std::vector<Communicate::request> send_requests;
  std::vector<Communicate::request> recv_requests;
  const int exchange_tag = 27183;
  using ExchangeType = BsplineExchangeData<PointType>;

  // 1 value + 3 gradient + 9 hessian
  static const int max_values_per_orbital = 13;
  int max_threads = 0;

  
  inline VectorViewer<value_type>
  sendValues(int rank, int offset) 
  {
    return VectorViewer<value_type>(
      (*send_values_ptr)[max_threads*rank*max_values_per_orbital+offset],local_spos);
  }

  inline VectorViewer<grad_type>
  sendGradients(int rank, int offset) 
  {
    return VectorViewer<grad_type>(
      (grad_type*)(*send_values_ptr)[max_threads*rank*max_values_per_orbital+offset],local_spos);
  }

  inline VectorViewer<value_type>
  sendLaplacians(int rank, int offset) 
  {
    return VectorViewer<value_type>(
      (*send_values_ptr)[max_threads*rank*max_values_per_orbital+offset],local_spos);
  }

  inline VectorViewer<hess_type>
  sendHessians(int rank, int offset) 
  {
    return VectorViewer<hess_type>(
      (hess_type*)(*send_values_ptr)[max_threads*rank*max_values_per_orbital+offset],local_spos);
  }

  inline VectorViewer<value_type>
  recvValues(int rank, int offset) 
  {
    size_t size = (SplineAdoptor::dist_spo_offsets[rank+1] - 
                   SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<value_type>((*recv_values_ptr)[rank][offset], size);
  }

  inline VectorViewer<grad_type>
  recvGradients(int rank, int offset) 
  {
    size_t size = (SplineAdoptor::dist_spo_offsets[rank+1] - 
                   SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<grad_type>((grad_type*)(*recv_values_ptr)[rank][offset+1], size);
  }

  inline VectorViewer<value_type>
  recvLaplacians(int rank, int offset) 
  {
    size_t size = (SplineAdoptor::dist_spo_offsets[rank+1] - 
                   SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<value_type>((value_type*)(*recv_values_ptr)[rank][offset+4], size);
  }

  inline VectorViewer<hess_type>
  recvHessians(int rank, int offset) 
  {
    size_t size = (SplineAdoptor::dist_spo_offsets[rank+1] - 
                   SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<hess_type>((hess_type*)(*recv_values_ptr)[rank][offset+4], size);
  }

  inline VectorViewer<value_type>
  recvVGL(int rank, int offset)
  {
    size_t size = 5*(SplineAdoptor::dist_spo_offsets[rank+1] - 
                     SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<value_type>((*recv_values_ptr)[rank][offset], size);
  }

  inline VectorViewer<value_type>
  recvVGH(int rank, int offset)
  {
    size_t size = 13*(SplineAdoptor::dist_spo_offsets[rank+1] - 
                     SplineAdoptor::dist_spo_offsets[rank]);
    return VectorViewer<value_type>((*recv_values_ptr)[rank][offset], size);
  }


  // We use shared_ptr so that when this object is cloned, all clones
  // have a pointer to the same underlying container.  This allows us
  // to exchange data between threads.
  std::shared_ptr<std::vector<ExchangeType>> exchange_data_ptr;
  std::shared_ptr<std::vector<ExchangeType>> my_exchange_data_ptr;

  std::shared_ptr<ValueMatrix_t> send_values_ptr;
  std::shared_ptr<std::vector<ValueMatrix_t> > recv_values_ptr;


  int group_size=0;
  int group_rank=0;
  int local_spos = 0;

  inline const typename SplineAdoptor::PointType
  groupPosition(int rank) 
  {
    int ithread = omp_get_thread_num();
    return (*exchange_data_ptr)[rank*max_threads+ithread].position;
  }

  int num_exch = 0;
  int
  exchangeData(const ExchangeType& my_data)
  {
    int thread_num = omp_get_thread_num();
    int num_evals = 0;
    if (!group_size) {
      group_size = SplineAdoptor::dist_group_comm->size();
      group_rank = SplineAdoptor::dist_group_comm->rank();
    }

    (*my_exchange_data_ptr)[thread_num] = my_data;
    
#pragma omp barrier
#pragma omp master
    {
      SplineAdoptor::dist_group_comm->allgather(
        (char*)my_exchange_data_ptr->data(), (char*)exchange_data_ptr->data(), 
        omp_get_num_threads()*sizeof(ExchangeType));
    }
#pragma omp barrier

    // Count the total number of evaluations to be performed after
    // this exchange
    for (int rank=0; rank < group_size; ++rank) {
      for (int ithread=0; ithread < max_threads; ++ithread) {
        auto quant = (*exchange_data_ptr)[rank*max_threads+ithread].quantities;
        num_evals += (quant != ExchangeType::EvalQuantities::NONE);
      }
    }
    return num_evals;

  }


  void
  postRemoteReceives(int &recv_offset)
  {
    int num_columns[] = { 0, 1, 5, 13 };
    int mythread = omp_get_thread_num();

    // We assume all threads have registered their quantity
    recv_offset=0;
    int recv_total=0;
    for (int ithread=0; ithread < max_threads; ++ithread) {
      typename ExchangeType::EvalQuantities quant = 
        (*my_exchange_data_ptr)[ithread].quantities;
      int cols = num_columns[(int)quant];
      recv_total += cols;
      if (ithread < mythread) {
        recv_offset += cols;
      }
    }

    // Post receives first
#pragma omp master
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        if (recv_total) {
          size_t offset = SplineAdoptor::dist_spo_offsets[rank];
          size_t size   = SplineAdoptor::dist_spo_offsets[rank+1] - offset;

          VectorViewer<value_type> recv_buff((*recv_values_ptr)[rank][0], 
                                             recv_total*size);
#ifdef SEPARATE_WAITS
          recv_requests[rank] = 
            SplineAdoptor::dist_group_comm->irecv(rank, exchange_tag, recv_buff);
#else
          requests.push_back(
            SplineAdoptor::dist_group_comm->irecv(rank, exchange_tag, recv_buff));
#endif
        }
      }
    }
  }

  void
  waitOnTransfers()
  {
#pragma omp master
    {
      int num_requests = requests.size();
      if (statuses.size() != requests.size()) {
        statuses.resize(requests.size());
      }
      MPI_Waitall(num_requests, &(requests[0]), &(statuses[0]));
      requests.clear();
    }
#pragma omp barrier
  }


  void
  waitOnReceives()
  {
#pragma omp master
    {
      for (int rank=0; rank < group_size; ++rank) {
        if (rank != group_rank) {
          MPI_Status recv_wait_status;
          if (recv_requests[rank]) {
            MPI_Wait(&(recv_requests[rank]), &recv_wait_status);
            recv_requests[rank] = 0;
          }
        }
      }
    }

#pragma omp barrier
  }

  void
  waitOnSends()
  {
#pragma omp master 
    {
      for (int rank=0; rank < group_size; ++rank) {
        if (rank != group_rank) {
          MPI_Status send_wait_status;
          if (send_requests[rank]) {
            MPI_Wait(&(send_requests[rank]), &send_wait_status);
            send_requests[rank] = 0;
          }
        }
      }
    }

#pragma omp barrier
  }



public:


  /// Constructor allocates shared containers
  DistributedBsplineSet() {
    exchange_data_ptr    = std::make_shared<std::vector<ExchangeType>>();
    my_exchange_data_ptr = std::make_shared<std::vector<ExchangeType>>();
    send_values_ptr      = std::make_shared<ValueMatrix_t>();
    recv_values_ptr      = std::make_shared<std::vector<ValueMatrix_t> >();
  }


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
  evaluateForRemote()
  {
    int mythread = omp_get_thread_num();

    //#pragma omp barrier

    // Number of columns to send for NONE, V, VGL, and VGH, respectively
    int num_columns[] = { 0, 1, 5, 13 };

    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        // First evaluate the offset for each thread to allow packing
        // of data from each thread into a contiguous send buffer. 
        int send_offset=0, send_total=0;
        for (int ithread=0; ithread < max_threads; ++ithread) {
          typename ExchangeType::EvalQuantities quant = 
            (*exchange_data_ptr)[rank*max_threads+ithread].quantities;
          int cols = num_columns[(int)quant];
          send_total += cols;
          if (ithread < mythread) {
            send_offset += cols;
          }
        }
        
        typename ExchangeType::EvalQuantities quant = 
          (*exchange_data_ptr)[rank*max_threads+mythread].quantities;
        PointType pos = (*exchange_data_ptr)[rank*max_threads+mythread].position;

        switch (quant) {
        case ExchangeType::EvalQuantities::NONE:
          break;
        case ExchangeType::EvalQuantities::V:
          {
            // VectorViewer<value_type> v = sendValues(rank);
            VectorViewer<value_type> v = sendValues(rank, send_offset);
            SplineAdoptor::evaluate_v(groupPosition(rank), v);
          }
          break;
        case ExchangeType::EvalQuantities::VGL:
          {
            VectorViewer<value_type> v   = sendValues(rank, send_offset);
            VectorViewer<grad_type>  g   = sendGradients(rank, send_offset+1);
            VectorViewer<value_type> l   = sendLaplacians(rank, send_offset+4);
            SplineAdoptor::evaluate_vgl(groupPosition(rank), v, g, l);
          }
          break;
        case ExchangeType::EvalQuantities::VGH:
          {
            VectorViewer<value_type> v   = sendValues(rank, send_offset);
            VectorViewer<grad_type>  g   = sendGradients(rank, send_offset+1);
            VectorViewer<hess_type>  h   = sendHessians(rank, send_offset+4);
            SplineAdoptor::evaluate_vgh(groupPosition(rank), v, g, h);
          }
          break;
        default:
          fprintf(stderr, "Logic error, unknown quantity code in "
                  "DistributedBsplineSet::evaluateForRemote\n");
        }
        // Synchronize with other threads
#pragma omp barrier
        // Only master thread sends
#pragma omp master
        {
          if (send_total) {
            VectorViewer<value_type> send_view =
              VectorViewer<value_type>(
                (*send_values_ptr)[max_threads*rank*max_values_per_orbital],
                send_total*local_spos);
            
#ifdef SEPARATE_WAITS
            send_requests[rank] = 
              SplineAdoptor::dist_group_comm->isend(rank, exchange_tag, send_view);
#else
            requests.push_back(
              SplineAdoptor::dist_group_comm->isend(rank, exchange_tag, send_view));
#endif

          }
        }
        // Synchronize with other threads
#pragma omp barrier
      }
    }
  }

  
  int num_value_calls=0;
  inline void evaluate(const ParticleSet& P, int iat, ValueVector_t& psi)
  {
    exchangeData(ExchangeType(P.R[iat], ExchangeType::EvalQuantities::V));

    int my_recv_offset=0;
    // my_recv_offset will have the column offset in the receive
    // buffer where this thread's remote results are stored.
    postRemoteReceives(my_recv_offset);

    evaluateForRemote();

    // evaluate local part
    size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    VectorViewer<value_type> v(&(psi[offset]),local_spos);
    SplineAdoptor::evaluate_v(groupPosition(group_rank), v);

#ifdef SEPARATE_WAITS
    waitOnReceives();
#else
    waitOnTransfers();
#endif
    
    // Copy received values into the right place
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        size_t offset = SplineAdoptor::dist_spo_offsets[rank];
        size_t size   = SplineAdoptor::dist_spo_offsets[rank+1] - offset;
        VectorViewer<value_type> recv = recvValues(rank, my_recv_offset);
        simd::copy(&(psi[offset]), &(recv[0]), size);
      }
    }
#ifdef SEPARATE_WAITS
    waitOnSends();
#endif
  }

  inline void evaluateValues(const ParticleSet& P, ValueMatrix_t& psiM)
  {
    ValueVector_t psi(psiM.cols());
    for(int iat=0; iat<P.getTotalNum(); ++iat)
    {
      evaluate(P, iat, psi);
      std::copy(psi.begin(),psi.end(),psiM[iat]);
    }
  }

  int num_VGL_calls=0;
  inline void evaluate(const ParticleSet& P, int iat,
                       ValueVector_t& psi, GradVector_t& dpsi, ValueVector_t& d2psi)
  {
    exchangeData(ExchangeType(P.R[iat], ExchangeType::EvalQuantities::VGL));

    int my_recv_offset=0;
    postRemoteReceives(my_recv_offset);
    evaluateForRemote();

    // Evaluate local part
    size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    VectorViewer<value_type> v(&(  psi[offset]),local_spos);
    VectorViewer<grad_type>  g(&( dpsi[offset]),local_spos);
    VectorViewer<value_type> l(&(d2psi[offset]),local_spos);
    SplineAdoptor::evaluate_vgl(groupPosition(group_rank), v, g, l);

#ifdef SEPARATE_WAITS
    waitOnReceives();
#else
    waitOnTransfers();
#endif

    // Now copy received data to psi, dpsi, and d2psi
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        size_t offset = SplineAdoptor::dist_spo_offsets[rank];
        size_t size   = SplineAdoptor::dist_spo_offsets[rank+1] - offset;
        VectorViewer<value_type> v = recvValues(rank, my_recv_offset);
        VectorViewer<grad_type>  g = recvGradients(rank, my_recv_offset);
        VectorViewer<value_type> l = recvLaplacians(rank, my_recv_offset);
        simd::copy(&(  psi[offset]), &(v[0]), size);
        simd::copy(&( dpsi[offset]), &(g[0]), size);
        simd::copy(&(d2psi[offset]), &(l[0]), size);
      }
    }    
    
#ifdef SEPARATE_WAITS
    waitOnSends();
#endif
  }

  int num_VGH_calls=0;
  inline void evaluate(const ParticleSet& P, int iat,
                       ValueVector_t& psi, GradVector_t& dpsi, HessVector_t& grad_grad_psi)
  {
    exchangeData(ExchangeType(P.R[iat], ExchangeType::EvalQuantities::VGH));

    int my_recv_offset=0;
    postRemoteReceives(my_recv_offset);
    evaluateForRemote();

    // Local evaluation
    size_t offset = SplineAdoptor::dist_spo_offsets[group_rank];
    size_t size   = SplineAdoptor::dist_spo_offsets[group_rank+1] - offset;
    VectorViewer<value_type> v(&(  psi[offset]),local_spos);
    VectorViewer<grad_type>  g(&( dpsi[offset]),local_spos);
    VectorViewer<hess_type>  h(&(grad_grad_psi[offset]),local_spos);
    SplineAdoptor::evaluate_vgh(groupPosition(group_rank), v, g, h);

    // Wait for all receives to complete
#ifdef SEPARATE_WAITS
    waitOnReceives();
#else
    waitOnTransfers();
#endif

    // Now copy received data to psi, dpsi, and d2psi
    for (int rank=0; rank < group_size; ++rank) {
      if (rank != group_rank) {
        size_t offset = SplineAdoptor::dist_spo_offsets[rank];
        size_t size   = SplineAdoptor::dist_spo_offsets[rank+1] - offset;
        VectorViewer<value_type> v = recvValues(rank, my_recv_offset);
        VectorViewer<grad_type>  g = recvGradients(rank, my_recv_offset);
        VectorViewer<hess_type>  h = recvHessians(rank, my_recv_offset);
        simd::copy(&(  psi[offset]), &(v[0]), size);
        simd::copy(&( dpsi[offset]), &(g[0]), size);
        simd::copy(&(grad_grad_psi[offset]), &(h[0]), size);
      }
    }    
    
#ifdef SEPARATE_WAITS
    waitOnSends();
#endif
  }

  void resetParameters(const opt_variables_type& active)
  { }

  void resetTargetParticleSet(ParticleSet& e)
  { }

  void setOrbitalSetSize(int norbs)
  {
    OrbitalSetSize = norbs;
    BasisSetSize=norbs;
  }

  virtual void
  resizeRemoteStorage(size_t group_size, size_t num_local_spos) override
  {
    max_threads = omp_get_max_threads();
    local_spos = num_local_spos;
    
    int total_size = group_size * max_threads;
#pragma omp master
    {
      exchange_data_ptr->resize(total_size);
      my_exchange_data_ptr->resize(max_threads);
      send_values_ptr->resize(total_size*max_values_per_orbital, num_local_spos);
      recv_values_ptr->resize(group_size);
      send_requests.resize(group_size, 0);
      recv_requests.resize(group_size, 0);
      for (int rank=0; rank < group_size; ++rank) {
        if (rank != SplineAdoptor::dist_group_comm->rank()) {
          size_t num_orbs = SplineAdoptor::dist_spo_offsets[rank+1] -
            SplineAdoptor::dist_spo_offsets[rank];
          (*recv_values_ptr)[rank].resize(max_threads*max_values_per_orbital, num_orbs);
        }
      }
    }

#pragma omp barrier
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
    }
    completeDistributedEvaluations(0);
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
    }
    completeDistributedEvaluations(0);
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
    int num_evals=0;
    int recv_offset=0;
    do {
      num_evals = exchangeData(exch);
      postRemoteReceives(recv_offset);
      evaluateForRemote();
#ifdef SEPARATE_WAITS
      waitOnReceives();
      waitOnSends();
#else
      waitOnTransfers();
#endif
    } while (num_evals);
  }

  virtual Communicate*
  getDistributedOrbitalComm() const
  {
    return SplineAdoptor::dist_group_comm;
  }



};

}

#endif
