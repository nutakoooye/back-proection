
def test_main(client_values, consortPath:str):
    Kss = client_values['Kss']
    dxsint = client_values['dxsint']
    dysint = client_values['dysint']
    StepBright = client_values['StepBright']
    Nxsint = client_values['Nxsint']
    Nysint = client_values['Nysint']
    Tsint = client_values['Tsint']
    tauRli = client_values['tauRli']
    RegimRsa = client_values['RegimRsa']
    TypeWinDp = client_values['TypeWinDp']
    TypeWinDn = client_values['TypeWinDn']
    GPUCalculationFlag = client_values['isGPU']
    FlagViewSignal = client_values['FlagViewSignal']



# 't_r_w': float(ui.findChild(QDoubleSpinBox, "t_r_w").text().replace(',', '.')),  # мб не то :)