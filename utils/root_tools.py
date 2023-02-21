import ROOT

def tset(tobj, **attrs):
    for attr, value in attrs.items():
        setter = getattr(tobj, f'Set{attr}', None)
        if setter is None: 
            print(f'[WARNING] unable to set attribute {attr} for {type(tobj)}')
            continue
        if not isinstance(value, (tuple, list)): value = [value]
        setter(*value)
    return tobj
    
def format_histo(name, title, bins, start, end, color=ROOT.kBlack, **set_attrs):
    histo = ROOT.TH1F(name, title, bins, start, end)
    histo.SetLineColor(color)
    histo.SetMarkerColor(color)
    histo.SetMarkerStyle(20)
    histo.Sumw2()

    return tset(histo, **set_attrs)


def format_histo2d(name, title, bins, start, end, bins2, start2, end2, color=ROOT.kBlack, **set_attrs):
    histo = ROOT.TH2F(name, title, bins, start, end, bins2, start2, end2)
    histo.SetLineColor(color)
    histo.SetMarkerColor(color)
    histo.SetMarkerStyle(20)
    histo.Sumw2()

    return tset(histo, **set_attrs)