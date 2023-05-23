def getFilesPath(consortPath):
    ConsortPath = consortPath.replace("\\", "/")
    arr = ConsortPath.split('/')
    arr[-1] = arr[-1].replace('Consort', 'ModelDate')
    ModelDatePath = '/'.join(arr)
    arr[-1] = arr[-1].replace('ModelDate', 'Yts1').replace('.txt', '.bin')
    Yts1Path = '/'.join(arr)
    arr[-1] = arr[-1].replace('Yts1', 'Yts2')
    Yts2Path = '/'.join(arr)
    result = ConsortPath, ModelDatePath, Yts1Path, Yts2Path
    return result
