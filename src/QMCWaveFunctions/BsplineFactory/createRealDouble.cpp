//////////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the University of Illinois/NCSA Open Source License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2016 Jeongnim Kim and QMCPACK developers.
//
// File developed by:
//
// File created by: Jeongnim Kim, jeongnim.kim@intel.com, Intel Corp.
//////////////////////////////////////////////////////////////////////////////////////
#include "QMCWaveFunctions/BsplineFactory/macro.h"
#include "qmc_common.h"
#include <Utilities/ProgressReportEngine.h>
#include "QMCWaveFunctions/EinsplineSetBuilder.h"
#include "QMCWaveFunctions/EinsplineAdoptor.h"
#include "QMCWaveFunctions/DistributedBsplineSet.h"
#include "QMCWaveFunctions/SplineR2RAdoptor.h"
#include "QMCWaveFunctions/BsplineFactory/SplineR2RAdoptor.h"
#include "QMCWaveFunctions/BsplineFactory/HybridRealAdoptor.h"
#include <fftw3.h>
#include <QMCWaveFunctions/einspline_helper.hpp>
#include "QMCWaveFunctions/BsplineReaderBase.h"
#include "QMCWaveFunctions/SplineAdoptorReaderP.h"
#include "QMCWaveFunctions/SplineHybridAdoptorReaderP.h"

namespace qmcplusplus
{

  BsplineReaderBase* 
  createBsplineRealDouble(EinsplineSetBuilder* e, 
                          bool hybrid_rep, int dist_group_size)
  {
    BsplineReaderBase* aReader=nullptr;
#if defined(QMC_ENABLE_SOA_DET)
    if(hybrid_rep) {
      aReader= new SplineHybridAdoptorReader<HybridRealSoA<SplineR2RSoA<double,OHMMS_PRECISION> > >(e);
    }
    else if (dist_group_size != 1) {
      auto reader = new SplineAdoptorReader<SplineR2RSoA<double,OHMMS_PRECISION>,true >(e);
      reader->dist_group_size = dist_group_size;
      aReader = reader;
    }
    else {
      aReader= new SplineAdoptorReader<SplineR2RSoA<double,OHMMS_PRECISION>,false >(e);
    }
#else
    if (dist_group_size != 1) {
      auto reader= new SplineAdoptorReader<SplineR2RAdoptor<double,OHMMS_PRECISION,3>,true >(e);
      reader->dist_group_size = dist_group_size;
      aReader = reader;
    }
    else {
      aReader= new SplineAdoptorReader<SplineR2RAdoptor<double,OHMMS_PRECISION,3>,false >(e);
    }
#endif
    return aReader;
  }
}
