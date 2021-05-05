/* 6b Preselection
 * Author: Suzanne Rosenzweig, s.rosenzweig@cern.ch
 */

#include <iostream>
#include "TH1F.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TChain.h"
#include <cstdlib>
#include <vector>
#include "TString.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

int apply_preselections(){

  TFile *fout = new TFile("signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_reco_preselections.root","RECREATE");
  TTree *t1   = new TTree("sixBtree","sixBtree");

  TString tree = "sixBtree";
  TChain *cc  = new TChain(tree);

  TString file1 = "signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_accstudies.root";
  TString file2 = "signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_accstudies_2.root";
  TString file3 = "signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_accstudies_500k.root";
  TString file4 = "signal/NanoAOD/NMSSM_XYH_YToHH_6b_MX_700_MY_400_accstudies_500k_2.root";

  cc->Add(file1);
  cc->Add(file2);
  cc->Add(file3);
  cc->Add(file4);

  TTreeReader reader(cc);
        
  TTreeReaderValue<float> gen_HX_b1_recojet_m(reader,"gen_HX_b1_recojet_m");
  TTreeReaderValue<float> gen_HX_b1_recojet_pt(reader,"gen_HX_b1_recojet_pt");
  TTreeReaderValue<float> gen_HX_b1_recojet_eta(reader,"gen_HX_b1_recojet_eta");
  TTreeReaderValue<float> gen_HX_b1_recojet_phi(reader,"gen_HX_b1_recojet_phi");
  TTreeReaderValue<float> gen_HX_b1_recojet_ptRegressed(reader,"gen_HX_b1_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HX_b2_recojet_m(reader,"gen_HX_b2_recojet_m");
  TTreeReaderValue<float> gen_HX_b2_recojet_pt(reader,"gen_HX_b2_recojet_pt");
  TTreeReaderValue<float> gen_HX_b2_recojet_eta(reader,"gen_HX_b2_recojet_eta");
  TTreeReaderValue<float> gen_HX_b2_recojet_phi(reader,"gen_HX_b2_recojet_phi");
  TTreeReaderValue<float> gen_HX_b2_recojet_ptRegressed(reader,"gen_HX_b2_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HY1_b1_recojet_m(reader,"gen_HY1_b1_recojet_m");
  TTreeReaderValue<float> gen_HY1_b1_recojet_pt(reader,"gen_HY1_b1_recojet_pt");
  TTreeReaderValue<float> gen_HY1_b1_recojet_eta(reader,"gen_HY1_b1_recojet_eta");
  TTreeReaderValue<float> gen_HY1_b1_recojet_phi(reader,"gen_HY1_b1_recojet_phi");
  TTreeReaderValue<float> gen_HY1_b1_recojet_ptRegressed(reader,"gen_HY1_b1_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HY1_b2_recojet_m(reader,"gen_HY1_b2_recojet_m");
  TTreeReaderValue<float> gen_HY1_b2_recojet_pt(reader,"gen_HY1_b2_recojet_pt");
  TTreeReaderValue<float> gen_HY1_b2_recojet_eta(reader,"gen_HY1_b2_recojet_eta");
  TTreeReaderValue<float> gen_HY1_b2_recojet_phi(reader,"gen_HY1_b2_recojet_phi");
  TTreeReaderValue<float> gen_HY1_b2_recojet_ptRegressed(reader,"gen_HY1_b2_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HY2_b1_recojet_m(reader,"gen_HY2_b1_recojet_m");
  TTreeReaderValue<float> gen_HY2_b1_recojet_pt(reader,"gen_HY2_b1_recojet_pt");
  TTreeReaderValue<float> gen_HY2_b1_recojet_eta(reader,"gen_HY2_b1_recojet_eta");
  TTreeReaderValue<float> gen_HY2_b1_recojet_phi(reader,"gen_HY2_b1_recojet_phi");
  TTreeReaderValue<float> gen_HY2_b1_recojet_ptRegressed(reader,"gen_HY2_b1_recojet_ptRegressed");

  TTreeReaderValue<float> gen_HY2_b2_recojet_m(reader,"gen_HY2_b2_recojet_m");
  TTreeReaderValue<float> gen_HY2_b2_recojet_pt(reader,"gen_HY2_b2_recojet_pt");
  TTreeReaderValue<float> gen_HY2_b2_recojet_eta(reader,"gen_HY2_b2_recojet_eta");
  TTreeReaderValue<float> gen_HY2_b2_recojet_phi(reader,"gen_HY2_b2_recojet_phi");
  TTreeReaderValue<float> gen_HY2_b2_recojet_ptRegressed(reader,"gen_HY2_b2_recojet_ptRegressed");

  float HX_b1_recojet_m, HX_b1_recojet_pt, HX_b1_recojet_ptRegressed, HX_b1_recojet_eta, HX_b1_recojet_phi,  HX_b2_recojet_m, HX_b2_recojet_pt, HX_b2_recojet_ptRegressed, HX_b2_recojet_eta, HX_b2_recojet_phi, HY1_b1_recojet_m, HY1_b1_recojet_pt, HY1_b1_recojet_ptRegressed, HY1_b1_recojet_eta, HY1_b1_recojet_phi,  HY1_b2_recojet_m, HY1_b2_recojet_pt, HY1_b2_recojet_ptRegressed, HY1_b2_recojet_eta, HY1_b2_recojet_phi, HY2_b1_recojet_m, HY2_b1_recojet_pt, HY2_b1_recojet_ptRegressed, HY2_b1_recojet_eta, HY2_b1_recojet_phi,  HY2_b2_recojet_m, HY2_b2_recojet_pt, HY2_b2_recojet_ptRegressed, HY2_b2_recojet_eta, HY2_b2_recojet_phi;

  t1->Branch("HX_b1_recojet_m",& HX_b1_recojet_m);
  t1->Branch("HX_b1_recojet_pt",& HX_b1_recojet_pt);
  t1->Branch("HX_b1_recojet_eta",& HX_b1_recojet_eta);
  t1->Branch("HX_b1_recojet_phi",& HX_b1_recojet_phi);
  t1->Branch("HX_b1_recojet_ptRegressed",& HX_b1_recojet_ptRegressed);

  t1->Branch("HX_b2_recojet_m",& HX_b2_recojet_m);
  t1->Branch("HX_b2_recojet_pt",& HX_b2_recojet_pt);
  t1->Branch("HX_b2_recojet_eta",& HX_b2_recojet_eta);
  t1->Branch("HX_b2_recojet_phi",& HX_b2_recojet_phi);
  t1->Branch("HX_b2_recojet_ptRegressed",& HX_b2_recojet_ptRegressed);

  t1->Branch("HY1_b1_recojet_m",& HY1_b1_recojet_m);
  t1->Branch("HY1_b1_recojet_pt",& HY1_b1_recojet_pt);
  t1->Branch("HY1_b1_recojet_eta",& HY1_b1_recojet_eta);
  t1->Branch("HY1_b1_recojet_phi",& HY1_b1_recojet_phi);
  t1->Branch("HY1_b1_recojet_ptRegressed",& HY1_b1_recojet_ptRegressed);

  t1->Branch("HY1_b2_recojet_m",& HY1_b2_recojet_m);
  t1->Branch("HY1_b2_recojet_pt",& HY1_b2_recojet_pt);
  t1->Branch("HY1_b2_recojet_eta",& HY1_b2_recojet_eta);
  t1->Branch("HY1_b2_recojet_phi",& HY1_b2_recojet_phi);
  t1->Branch("HY1_b2_recojet_ptRegressed",& HY1_b2_recojet_ptRegressed);

  t1->Branch("HY2_b1_recojet_m",& HY2_b1_recojet_m);
  t1->Branch("HY2_b1_recojet_pt",& HY2_b1_recojet_pt);
  t1->Branch("HY2_b1_recojet_eta",& HY2_b1_recojet_eta);
  t1->Branch("HY2_b1_recojet_phi",& HY2_b1_recojet_phi);
  t1->Branch("HY2_b1_recojet_ptRegressed",& HY2_b1_recojet_ptRegressed);

  t1->Branch("HY2_b2_recojet_m",& HY2_b2_recojet_m);
  t1->Branch("HY2_b2_recojet_pt",& HY2_b2_recojet_pt);
  t1->Branch("HY2_b2_recojet_eta",& HY2_b2_recojet_eta);
  t1->Branch("HY2_b2_recojet_phi",& HY2_b2_recojet_phi);
  t1->Branch("HY2_b2_recojet_ptRegressed",& HY2_b2_recojet_ptRegressed);

  int pt_threshold = 30;
  float eta_threshold = 2.4;

  int eventCount = 0;
  int passCount = 0;
  while(reader.Next()){
    eventCount++;
    if (eventCount % 10000 == 0) {
      std::cout << eventCount << " events read!" << std::endl;
    }

    if (*gen_HX_b1_recojet_ptRegressed < pt_threshold) continue;
    if (*gen_HX_b2_recojet_ptRegressed < pt_threshold) continue;
    if (*gen_HY1_b1_recojet_ptRegressed < pt_threshold) continue;
    if (*gen_HY1_b2_recojet_ptRegressed < pt_threshold) continue;
    if (*gen_HY2_b1_recojet_ptRegressed < pt_threshold) continue;
    if (*gen_HY2_b2_recojet_ptRegressed < pt_threshold) continue;

    if (abs(*gen_HX_b1_recojet_eta) > eta_threshold) continue;
    if (abs(*gen_HX_b2_recojet_eta) > eta_threshold) continue;
    if (abs(*gen_HY1_b1_recojet_eta) > eta_threshold) continue;
    if (abs(*gen_HY1_b2_recojet_eta) > eta_threshold) continue;
    if (abs(*gen_HY2_b1_recojet_eta) > eta_threshold) continue;
    if (abs(*gen_HY2_b2_recojet_eta) > eta_threshold) continue;

    passCount++;

    HX_b1_recojet_m   = *gen_HX_b1_recojet_m;
    HX_b1_recojet_pt  = *gen_HX_b1_recojet_pt;
    HX_b1_recojet_eta = *gen_HX_b1_recojet_eta;
    HX_b1_recojet_phi = *gen_HX_b1_recojet_phi;
    HX_b1_recojet_ptRegressed = *gen_HX_b1_recojet_ptRegressed;

    HX_b2_recojet_m   = *gen_HX_b2_recojet_m;
    HX_b2_recojet_pt  = *gen_HX_b2_recojet_pt;
    HX_b2_recojet_eta = *gen_HX_b2_recojet_eta;
    HX_b2_recojet_phi = *gen_HX_b2_recojet_phi;
    HX_b2_recojet_ptRegressed = *gen_HX_b2_recojet_ptRegressed;

    HY1_b1_recojet_m   = *gen_HY1_b1_recojet_m;
    HY1_b1_recojet_pt  = *gen_HY1_b1_recojet_pt;
    HY1_b1_recojet_eta = *gen_HY1_b1_recojet_eta;
    HY1_b1_recojet_phi = *gen_HY1_b1_recojet_phi;
    HY1_b1_recojet_ptRegressed = *gen_HY1_b1_recojet_ptRegressed;

    HY1_b2_recojet_m   = *gen_HY1_b2_recojet_m;
    HY1_b2_recojet_pt  = *gen_HY1_b2_recojet_pt;
    HY1_b2_recojet_eta = *gen_HY1_b2_recojet_eta;
    HY1_b2_recojet_phi = *gen_HY1_b2_recojet_phi;
    HY1_b2_recojet_ptRegressed = *gen_HY1_b2_recojet_ptRegressed;

    HY2_b1_recojet_m   = *gen_HY2_b1_recojet_m;
    HY2_b1_recojet_pt  = *gen_HY2_b1_recojet_pt;
    HY2_b1_recojet_eta = *gen_HY2_b1_recojet_eta;
    HY2_b1_recojet_phi = *gen_HY2_b1_recojet_phi;
    HY2_b1_recojet_ptRegressed = *gen_HY2_b1_recojet_ptRegressed;

    HY2_b2_recojet_m   = *gen_HY2_b2_recojet_m;
    HY2_b2_recojet_pt  = *gen_HY2_b2_recojet_pt;
    HY2_b2_recojet_eta = *gen_HY2_b2_recojet_eta;
    HY2_b2_recojet_phi = *gen_HY2_b2_recojet_phi;
    HY2_b2_recojet_ptRegressed = *gen_HY2_b2_recojet_ptRegressed;
    
    t1->Fill();

  } // end event loop
  
  t1->Write();

  std::cout << passCount << " Events Passed Preselections" << endl;

  return 0;
} // end function

