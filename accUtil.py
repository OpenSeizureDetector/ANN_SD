

def normaliseAccArr(dataArr):
    """
    Normalise a numpy vector values.   Sets zero to be the main
    of the data, and normalises to +/-1 based on a 2000 mG assumed
    full scale deflection of the instrument.
    """
    FSD = 2000  # Assumed full scale deflection in mG.
    # Normalise the data
    dataArr = dataArr.astype('float32') - dataArr.mean()
    dataArr = dataArr / FSD
    return(dataArr)
