% формирование ЛЧМ-сигнала оо
function sigLFM = ChirpSig(t,tx,b)
j=sqrt(-1);
win=(t>=0)&(t<=tx);
sigLFM=win*exp(j*b*t^2);

