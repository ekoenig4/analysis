from ... import *
from ... import eightbUtils as eightb

class TestAnalysis(Analysis):
    def plot_n1_histos(self):
        study.h_quick( 
            self.signal[self.use_signal]+self.bkg,
            legend=True,
            plot_scale=[100]*len(self.use_signal),
            varlist=['n_ele','n_mu','n_presel_jet'],
            saveas=f'n1_histos',
        )

    def reweight_bkg(self):
        self.bkg.reweight(2.3)
        self.bkg.set_systematics(0.2)
    
    def jet_kin_cuts(self):
        ptcut = eightb.selected_jet_pt()
        btagcut = eightb.selected_jet_btagwp()

        event_filter = FilterSequence(
            ptcut, btagcut
        )

        self.signal = self.signal.apply(event_filter)
        self.bkg = self.bkg.apply(event_filter)
        self.data = self.data.apply(event_filter)

    def cutflow(self):
        cutflow_labels = [
            "total", "trigger","met filters", "muon veto", "electron veto", "n_presel_jets >= 8", "selected_jets", "selected_jets_pt", "selected_jets_btag"
        ]

        study.cutflow( 
            self.signal[self.use_signal] + self.bkg + self.data,
            ylim=(1e3, -1),
            xlabel=cutflow_labels,
            legend=True,
            saveas=f'{self.dout}/cutflow'
        )

    def plot_signal(self):
        study.quick( 
            self.signal[self.use_signal],
            legend=True,
            varlist=['nfound_select','nfound_paired_h'],
            xlabels=['N Higgs Jet','N Paired Higgs'],
            saveas=f'{self.dout}/signal_n_reco'
        )

    def plot_kin(self):
        for var in ('pt','btag','eta','phi',):
            study.quick( 
                self.signal[self.use_signal]+self.bkg,
                legend=True,
                plot_scale=[10]*len(self.use_signal),
                varlist=[f'jet_{var}[:,{i}]'  for i in range(8) ],
                h_rebin=20,
                dim=(-1, 4),
                saveas=f'{self.dout}/kin/jet_kin_by_{var}'
            )
        def renaming_variables(t):
            y1_p4 = build_p4(t, prefix='Y1')
            y2_p4 = build_p4(t, prefix='Y2')
            t.extend(
                higgs_jet_dr=t.higgs_dr,
                Y1_higgs_dr=t.Y1_dr,
                Y2_higgs_dr=t.Y2_dr,
                X_y_dr=calc_dr_p4(y1_p4, y2_p4),
            )
        (self.signal+self.bkg+self.data).apply(renaming_variables)

        for var in ('pt','m','jet_dr','eta'):
            study.quick( 
                self.signal[self.use_signal]+self.bkg,
                legend=True,
                plot_scale=[10]*len(self.use_signal),
                varlist=[f'higgs_{var}[:,{i}]' for i in range(4)],
                h_rebin=20,
                dim=(-1, 4),
                saveas=f'{self.dout}/kin/higgs_kin_by_{var}'
            )
        
        for var in ('pt','m','higgs_dr','eta'):
            study.quick( 
                self.signal[self.use_signal]+self.bkg,
                legend=True,
                plot_scale=[10]*len(self.use_signal),
                varlist=[f'Y{i+1}_{var}' for i in range(2) ],
                h_rebin=20,
                dim=(-1, 2),
                saveas=f'{self.dout}/kin/y_kin_by_{var}'
            )

        study.quick( 
            self.signal[self.use_signal]+self.bkg,
            legend=True,
            plot_scale=[10]*len(self.use_signal),
            varlist=[f'X_{var}' for var in ('pt','m','y_dr','eta')],
            h_rebin=20,
            dim=(-1, 4),
            saveas=f'{self.dout}/kin/X_kin'
        )

    def plot_2d_higgs_m(self):
        study.quick2d(
            self.signal[self.use_signal]+self.bkg,
            xvarlist=['higgs_m[:,0]',],
            yvarlist=['higgs_m[:,1]',],
            binlist=[(0,400,30)]*2,
            exe=[
                draw_concentric(*self.ar_center[:2], self.sr_r, self.cr_r, label='- AR -\nTotal: {total_count:0.2e}({total_eff:0.2%})\nSR: {inner_count:0.2e}({inner_eff:0.2%})\nCR: {outer_count:0.2e}({outer_eff:0.2%})', text=(0.05,0.9), linewidth=2),
                draw_concentric(*self.vr_center[:2], self.sr_r, self.cr_r, label='- VR -\nTotal: {total_count:0.2e}({total_eff:0.2%})\nSR: {inner_count:0.2e}({inner_eff:0.2%})\nCR: {outer_count:0.2e}({outer_eff:0.2%})', text=(0.5,0.9), linewidth=2),
            ],
            scatter=True,
            saveas=f'{self.dout}/2d_higgs_space'
        )

    def build_higgs_dm(self):
        def higgs_dm(t):
            dm = ak.zeros_like(t.Run)
            for i, m in enumerate(self.ar_center):
                dm = dm + ( t.higgs_m[:,i] - m )**2
            dm = np.sqrt(dm)
            t.extend(higgs_dm = dm)
        (self.signal+self.bkg+self.data).apply(higgs_dm)

        def val_higgs_dm(t):
            dm = ak.zeros_like(t.Run)
            for i, m in enumerate(self.vr_center):
                dm = dm + ( t.higgs_m[:,i] - m )**2
            dm = np.sqrt(dm)
            t.extend(val_higgs_dm = dm)
        (self.signal+self.bkg+self.data).apply(val_higgs_dm)

    def plot_abcd_regions(self):
        study.quick( 
            self.signal[self.use_signal]+self.bkg,
            legend=True, 
            plot_scale=[10]*len(self.use_signal),
            varlist=['n_medium_btag','higgs_dm'],
            binlist=[np.arange(4,10),(0, self.cr_r, 20)],
            saveas=f'{self.dout}/ar_abcd_variables',
        )

        study.quick2d( 
            self.signal[self.use_signal]+self.bkg,
            varlist=['n_medium_btag','higgs_dm'],
            binlist=[np.arange(4,10),(0, self.cr_r, 20)],
            exe=draw_abcd(x_r=(4,5,9), y_r=(0, self.sr_r, self.cr_r), regions=["C","D","A","B"]),
            saveas=f'{self.dout}/ar_abcd_2d_region',
        )

        study.quick( 
            self.signal[self.use_signal]+self.bkg,
            legend=True, 
            plot_scale=[10]*len(self.use_signal),
            varlist=['n_medium_btag','val_higgs_dm'],
            binlist=[np.arange(4,10),(0, self.cr_r, 20)],
            saveas=f'{self.dout}/vr_abcd_variables',
        )

        study.quick2d( 
            self.signal[self.use_signal]+self.bkg,
            varlist=['n_medium_btag','val_higgs_dm'],
            binlist=[np.arange(4,10),(0, self.cr_r, 20)],
            exe=draw_abcd(x_r=(4,5,9), y_r=(0, self.sr_r, self.cr_r), regions=["C","D","A","B"]),
            saveas=f'{self.dout}/vr_abcd_region',
        )

    def var_correlations(self):
        study.quick( 
            self.signal[self.use_signal] + self.bkg,
            legend=True,
            plot_scale=[10]*len(self.use_signal),
            varlist=['n_loose_btag','n_medium_btag','n_tight_btag'],
            dim=-1,
            saveas=f'{self.dout}/btag-multi'
        )

        study.quick2d( 
            self.signal[self.use_signal] + self.bkg,
            varlist=['higgs_m[:,0]','n_medium_btag'],
            dim=-1,
            saveas=f'{self.dout}/2d_higgs_m_vs_mbtag'
        )

        study.compare_masks( 
            self.signal[self.use_signal], self.bkg,
            h_color=None,
            legend=True,
            masks=[lambda t: t.n_medium_btag >=4, lambda t: t.n_medium_btag >=5, lambda t: t.n_medium_btag >=6],
            label=['N Medium Btag >= 4', 'N Medium Btag >= 5', 'N Medium Btag >= 6'],
            varlist=['higgs_m[:,0]'],
            efficiency=True,
            ratio=True, r_inv=True, 
            # r_o_smooth=True,

            saveas=f'{self.dout}/higgs_mass_vs_btagcut'
        )

    def build_abcd(self):
        self.ar_bdt = ABCD(
            features=self.bdt_features,
            a = lambda t : (t.n_medium_btag >  4) & (t.higgs_dm < self.sr_r),
            b = lambda t : (t.n_medium_btag <= 4) & (t.higgs_dm < self.sr_r),
            c = lambda t : (t.n_medium_btag >  4) & (t.higgs_dm > self.sr_r) & (t.higgs_dm < self.cr_r),
            d = lambda t : (t.n_medium_btag <= 4) & (t.higgs_dm > self.sr_r) & (t.higgs_dm < self.cr_r),
        )

        blind_filter = EventFilter('blinded', filter=lambda t : ~( self.ar_bdt.a(t) ))
        self.blinded_data  = self.data.apply(blind_filter)

        self.vr_bdt = ABCD(
            features=self.bdt_features,
            a = lambda t : (t.n_medium_btag >  4) & (t.val_higgs_dm < self.sr_r),
            b = lambda t : (t.n_medium_btag <= 4) & (t.val_higgs_dm < self.sr_r),
            c = lambda t : (t.n_medium_btag >  4) & (t.val_higgs_dm > self.sr_r) & (t.val_higgs_dm < self.cr_r),
            d = lambda t : (t.n_medium_btag <= 4) & (t.val_higgs_dm > self.sr_r) & (t.val_higgs_dm < self.cr_r),
        )
    
    def print_abcd_yields(self):
        print("----- Signal AR -----")
        for lines in zip(*[ self.ar_bdt.print_yields( s, lumi=2018, return_lines=True) for s in self.signal[self.use_signal]]):
            print(' | '.join(lines))

        print("----- Signal VR -----")
        for lines in zip(*[ self.vr_bdt.print_yields( s, lumi=2018, return_lines=True) for s in self.signal[self.use_signal]]):
            print(' | '.join(lines))
            
        print("----- Data/MC AR -----")
        for lines in zip( self.ar_bdt.print_yields( self.blinded_data, return_lines=True), self.ar_bdt.print_yields(self.bkg, lumi=2018, return_lines=True)):
            print(' | '.join(lines))

        print("----- Data/MC VR -----")
        for lines in zip(self.vr_bdt.print_yields(self.blinded_data, return_lines=True), self.vr_bdt.print_yields(self.bkg, lumi=2018, return_lines=True)):
            print(' | '.join(lines))

    def train_ar_bdt(self):
        self.bkg_model = self.blinded_data.asmodel('bkg model')

        # %%
        print("Training AR BDT")
        self.ar_bdt.train(self.bkg_model)
        self.ar_bdt.print_results(self.bkg_model)

    def calc_limits(self):
        study.quick(
            self.signal[self.use_signal] + self.bkg_model, 
            legend=True,
            masks=[self.ar_bdt.a]*len(self.use_signal)+[self.ar_bdt.b]*len(self.bkg_model),
            scale=[1]*len(self.use_signal)+[self.ar_bdt.reweight_tree]*len(self.bkg_model),
            varlist=['X_m'],
            binlist=[(500,2000,30)],
            title='BDT Bkg Model',
            limits=True,
            l_poi=np.linspace(0,3,21), 
            saveas=f'{self.dout}/limits/bdt_bkg_model'
        )

    def calc_brazil(self):
        study.brazil(
            self.signal+self.bkg_model,
            masks=[self.ar_bdt.a]*len(self.signal)+[self.ar_bdt.b]*len(self.bkg_model),
            scale=[1]*len(self.signal)+[self.ar_bdt.reweight_tree]*len(self.bkg_model),
            varlist=['X_m'],
            binlist=[(500,2000,30)],
            l_poi=np.linspace(0,3,21), 
            saveas=f'{self.dout}/limits/brazil_bdt_bkg_model'
        )

    def train_vr_bdt(self):
        self.vr_bdt.train(self.bkg_model)
        self.vr_bdt.print_results(self.bkg_model)

    def plot_vr_features(self):
        study.quick_region(
            self.blinded_data , self.bkg_model, self.bkg_model, label=['target','k factor','bdt'],
            h_color=['black','red','orange'], legend=True,
            masks=[self.vr_bdt.c]*len(self.blinded_data)+[self.vr_bdt.d]*(len(self.bkg_model)*2),
            scale=[1]*len(self.blinded_data)+[self.vr_bdt.scale_tree]*len(self.bkg_model)+[self.vr_bdt.reweight_tree]*len(self.bkg_model),
            varlist=self.vr_bdt.feature_names,
            h_rebin=15,
            suptitle='Validation Training (Data) D $\\rightarrow$ C',
            ratio=True,
            **study.kstest,
            saveas=f'{self.dout}/vr_bdt/training'
        )

        study.quick_region(
            self.blinded_data , self.bkg_model, self.bkg_model, label=['target','k factor','bdt'],
            h_color=['black','blue','green'], legend=True,
            masks=[self.vr_bdt.a]*len(self.blinded_data)+[self.vr_bdt.b]*(len(self.bkg_model)*2),
            scale=[1]*len(self.blinded_data)+[self.vr_bdt.scale_tree]*len(self.bkg_model)+[self.vr_bdt.reweight_tree]*len(self.bkg_model),
            varlist=self.vr_bdt.feature_names,
            h_rebin=15,
            suptitle='Validation Applying (Data) B $\\rightarrow$ A',
            ratio=True,
            **study.kstest,
            saveas=f'{self.dout}/vr_bdt/applying',
        )