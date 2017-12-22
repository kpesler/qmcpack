//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//                    Raymond Clay III, j.k.rofling@gmail.com, Lawrence Livermore National Laboratory
//                    Ye Luo, yeluo@anl.gov, Argonne National Laboratory
//                    Mark A. Berrill, berrillma@ornl.gov, Oak Ridge National Laboratory
//
// File created by: Jeongnim Kim, jeongnim.kim@gmail.com, University of Illinois at Urbana-Champaign
//////////////////////////////////////////////////////////////////////////////////////
    
    
/** @file BsplineReaderBase.h
 *
 * base class to read data and manage spline tables
 */
#ifndef QMCPLUSPLUS_BSPLINE_READER_BASE_H
#define QMCPLUSPLUS_BSPLINE_READER_BASE_H
#include <mpi/collectives.h>
#include <mpi/point2point.h>
namespace qmcplusplus
{

  class SPOSetInputInfo;

/**
 * Each SplineAdoptor needs a reader derived from BsplineReaderBase.
 * This base class handles common chores
 * - check_twists : read gvectors, set twists for folded bands if needed, and set the phase for the special K
 * - set_grid : create the basic grid and boundary conditions for einspline
 * Note that template is abused but it works.
 */
struct BsplineReaderBase
{
  ///pointer to the EinsplineSetBuilder
  EinsplineSetBuilder* mybuilder;
  ///communicator
  Communicate* myComm;
  ///mesh size
  TinyVector<int,3> MeshSize;
  ///first index of the SPO set
  int myFirstSPO;
  ///number of orbitals to be created
  int myNumOrbs;
  /**@addtogroup multigrid
   * @{
   * Currently only one level of coarsening is implemented but easy to generalize to multi levels.
   */
  /** dilation factor dense/corase only use one for all the directions
   */
  int GridFactor;
  ///cutoff radius for multigrid or other things
  double Rcut;
  /** @}*/
  ///map from spo index to band index
  std::vector<std::vector<int> > spo2band;

  BsplineReaderBase(EinsplineSetBuilder* e);

  virtual ~BsplineReaderBase();

  /** read gvectors and set the mesh, and prepare for einspline
   */
  template<typename GT, typename BCT>
  inline bool set_grid(const TinyVector<int,3>& halfg, GT* xyz_grid, BCT* xyz_bc)
  {
    //This sets MeshSize from the input file
    bool havePsig=mybuilder->ReadGvectors_ESHDF();

    //If this MeshSize is not initialized, use the meshsize set by the input based on FFT grid and meshfactor
    if(MeshSize[0]==0) MeshSize=mybuilder->MeshSize;

    app_log() << "  Using meshsize=" << MeshSize 
            << "\n  vs input meshsize=" << mybuilder->MeshSize << std::endl;

    xyz_grid[0].start = 0.0;
    xyz_grid[0].end = 1.0;
    xyz_grid[0].num = MeshSize[0];
    xyz_grid[1].start = 0.0;
    xyz_grid[1].end = 1.0;
    xyz_grid[1].num = MeshSize[1];
    xyz_grid[2].start = 0.0;
    xyz_grid[2].end = 1.0;
    xyz_grid[2].num = MeshSize[2];
    for(int j=0; j<3; ++j)
    {
      if(halfg[j])
      {
        xyz_bc[j].lCode=ANTIPERIODIC;
        xyz_bc[j].rCode=ANTIPERIODIC;
      }
      else
      {
        xyz_bc[j].lCode=PERIODIC;
        xyz_bc[j].rCode=PERIODIC;
      }
    }
    return havePsig;
  }

  /** initialize twist-related data for N orbitals
   */
  template<typename SPE>
  inline void check_twists(SPE* bspline, const BandInfoGroup& bandgroup)
  {
    //init(orbitalSet,bspline);
    bspline->PrimLattice=mybuilder->PrimCell;
    bspline->SuperLattice=mybuilder->SuperCell;
    bspline->GGt=mybuilder->GGt;
    //bspline->HalfG=mybuilder->HalfG;

    int N = bandgroup.getNumDistinctOrbitals();
    int numOrbs=bandgroup.getNumSPOs();

    bspline->my_unique_begin = 0;
    bspline->my_unique_end   = N;
    if (bspline->dist_group_comm) {
      int dist_group_size = bspline->dist_group_comm->size();
      int dist_rank       = bspline->dist_group_comm->rank();
      bspline->my_unique_begin = dist_rank*N / dist_group_size;
      bspline->my_unique_end   = (dist_rank+1)*N / dist_group_size;
      bspline->dist_spo_offsets.resize(dist_group_size+1);
    }

    int myNumUnique = bspline->my_unique_end - bspline->my_unique_begin;

    bspline->setOrbitalSetSize(numOrbs);
    bspline->resizeStorage(myNumUnique, myNumUnique);

    bspline->first_spo=bandgroup.getFirstSPO();
    bspline->last_spo =bandgroup.getLastSPO();

    int num = 0;
    const std::vector<BandInfo>& cur_bands=bandgroup.myBands;
    int my_first_spo = 0;
    for (int iorb=0; iorb<N; iorb++)
    {
      int ti = cur_bands[iorb].TwistIndex;

      if (iorb == bspline->my_unique_begin) {
        my_first_spo = num;
      }

      bool make_two_copies = cur_bands[iorb].MakeTwoCopies && (num<(numOrbs-1));
      if (iorb >= bspline->my_unique_begin && iorb < bspline->my_unique_end) {
        bspline->kPoints[iorb-bspline->my_unique_begin] = mybuilder->PrimCell.k_cart(mybuilder->TwistAngles[ti]); //twist);
        bspline->MakeTwoCopies[iorb-bspline->my_unique_begin] = make_two_copies;
      }
      num += make_two_copies ? 2 : 1;
    }

    // Make sure all bspline instances know which rank handles which SPOs
    if (bspline->dist_group_comm) {
      bspline->dist_spo_offsets[bspline->dist_group_comm->rank()] = my_first_spo;
      std::vector<int> send_buf(1);
      send_buf[0] = my_first_spo;
      bspline->dist_group_comm->allgather(send_buf, bspline->dist_spo_offsets, 1);
      // Set last offset to the total number of SPOs
      bspline->dist_spo_offsets[bspline->dist_group_comm->size()] = numOrbs;
      for (int i=0; i< bspline->dist_group_comm->size(); ++i) {
        app_log() << "Group rank " << i << " computes orbitals [" <<
          bspline->dist_spo_offsets[i] << "," <<  
          bspline->dist_spo_offsets[i+1]  << ")\n";
      }
      size_t group_rank = bspline->dist_group_comm->rank();
      size_t group_size = bspline->dist_group_comm->size();
      size_t num_local_spos = (bspline->dist_spo_offsets[group_rank+1] - 
                               bspline->dist_spo_offsets[group_rank+0]);
      bspline->resizeRemoteStorage(group_size, num_local_spos);
    }

    app_log() << "NumDistinctOrbitals " << N << " numOrbs = " << numOrbs << std::endl;

    bspline->HalfG=0;
    TinyVector<int,3> bconds=mybuilder->TargetPtcl.Lattice.BoxBConds;
    if(!bspline->is_complex)
    {
      //no k-point folding, single special k point (G, L ...)
      //TinyVector<double,3> twist0 = mybuilder->TwistAngles[cur_bands[0].TwistIndex];
      TinyVector<double,3> twist0 = mybuilder->TwistAngles[bandgroup.TwistIndex];
      for (int i=0; i<3; i++)
        if (bconds[i] && ((std::abs(std::abs(twist0[i]) - 0.5) < 1.0e-8)))
          bspline->HalfG[i] = 1;
        else
          bspline->HalfG[i] = 0;
      app_log() << "  TwistIndex = " << cur_bands[0].TwistIndex << " TwistAngle " << twist0 << std::endl;
      app_log() <<"   HalfG = " << bspline->HalfG << std::endl;
    }
    app_log().flush();
  }

  /** return the path name in hdf5
   */
  inline std::string psi_g_path(int ti, int spin, int ib)
  {
    std::ostringstream path;
    path << "/electrons/kpoint_" << ti
         << "/spin_" << spin << "/state_" << ib << "/psi_g";
    return path.str();
  }

  /** return the path name in hdf5
   */
  inline std::string psi_r_path(int ti, int spin, int ib)
  {
    std::ostringstream path;
    path << "/electrons/kpoint_" << ti
         << "/spin_" << spin << "/state_" << ib << "/psi_r";
    return path.str();
  }

  /** read/bcast psi_g
   * @param ti twist index
   * @param spin spin index
   * @param ib band index
   * @param cG psi_g as stored in hdf5
   */
  void get_psi_g(int ti, int spin, int ib, Vector<std::complex<double> >& cG);

  /** create the actual spline sets
   */
  virtual SPOSetBase* create_spline_set(int spin, const BandInfoGroup& bandgroup)=0;

  /** setting common parameters
   */
  void setCommon(xmlNodePtr cur);

  /** create the spline after one of the kind is created */
  SPOSetBase* create_spline_set(int spin, xmlNodePtr cur, SPOSetInputInfo& input_info);

  /** create the spline set */
  SPOSetBase* create_spline_set(int spin, xmlNodePtr cur);

  void initialize_spo2band(int spin, const std::vector<BandInfo>& bigspace, SPOSetInfo& sposet,
      std::vector<int>& band2spo);

  /** export the MultiSpline to the old class EinsplineSetExtended for the GPU calculation*/
  virtual void export_MultiSpline(multi_UBspline_3d_z **target)=0;
  virtual void export_MultiSpline(multi_UBspline_3d_d **target)=0;
};

}
#endif
