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
#include "Numerics/e2iphi.h"
#include "simd/vmath.hpp"
#include "qmc_common.h"
#include <Utilities/ProgressReportEngine.h>
#include "QMCWaveFunctions/EinsplineSetBuilder.h"
#include "QMCWaveFunctions/EinsplineAdoptor.h"
#include "QMCWaveFunctions/DistributedBsplineSet.h"
#include "QMCWaveFunctions/SplineC2XAdoptor.h"
#if defined(QMC_ENABLE_SOA_DET)
#include "QMCWaveFunctions/BsplineFactory/SplineC2RAdoptor.h"
#include "QMCWaveFunctions/BsplineFactory/SplineC2CAdoptor.h"
#include "QMCWaveFunctions/BsplineFactory/HybridCplxAdoptor.h"
#endif
#include <fftw3.h>
#include <QMCWaveFunctions/einspline_helper.hpp>
#include "QMCWaveFunctions/BsplineReaderBase.h"
#include "QMCWaveFunctions/SplineAdoptorReaderP.h"
#include "QMCWaveFunctions/SplineHybridAdoptorReaderP.h"

namespace qmcplusplus
{

  BsplineReaderBase* 
  createBsplineComplexSingle(EinsplineSetBuilder* e, bool hybrid_rep, 
                             bool distributed)
  {
    typedef OHMMS_PRECISION RealType;
    BsplineReaderBase* aReader=nullptr;

#if defined(QMC_COMPLEX)
  #if defined(QMC_ENABLE_SOA_DET)
    if(hybrid_rep)
      aReader= new SplineHybridAdoptorReader<HybridCplxSoA<SplineC2CSoA<float,RealType> > >(e);
    else if (distributed) {
      aReader= new SplineAdoptorReader<SplineC2CSoA<float,RealType>,true >(e);
    }
    else {
      aReader= new SplineAdoptorReader<SplineC2CSoA<float,RealType>,false >(e);
    }      
  #else
    if (distributed) {
      aReader= new SplineAdoptorReader<SplineC2CPackedAdoptor<float,RealType,3>,true >(e);
    }
    else {
      aReader= new SplineAdoptorReader<SplineC2CPackedAdoptor<float,RealType,3>,false >(e);      
    }
else
  #endif
#else //QMC_COMPLEX

  #if defined(QMC_ENABLE_SOA_DET)
    if(hybrid_rep) {
      aReader= new SplineHybridAdoptorReader<HybridCplxSoA<SplineC2RSoA<float,RealType> > >(e);
    }
    else if (distributed) {
      aReader= new SplineAdoptorReader<SplineC2RSoA<float,RealType>,true >(e);
    }
    else {
      aReader= new SplineAdoptorReader<SplineC2RSoA<float,RealType>,false >(e);
    }
  #else 
    if (distributed) {
      aReader= new SplineAdoptorReader<SplineC2RPackedAdoptor<float,RealType,3>, true >(e);
    }
    else {
      aReader= new SplineAdoptorReader<SplineC2RPackedAdoptor<float,RealType,3>, false >(e);
    }
  #endif
#endif

    return aReader;
  }
}

