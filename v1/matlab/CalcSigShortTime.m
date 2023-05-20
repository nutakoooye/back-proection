% скрипт для расчета траекторного сигнала для одного совмещенного канала
% или двух интерферометрических приемных каналов и одного передающего
% канала; расчет сигнала через СЖАТЫЙ, на выходе СПЕКТР
t=clock; disp(['Расчет дальностей ',num2str(t(1,4)),'.',num2str(t(1,5)),'.',num2str(t(1,6)),'.' ]);

% расчет для одного совмещенного канала 
if FlagInter==0
   % расчет дальностей   
   RT1=zeros(K,Q+1);
   for q=1:Q+1
       t=(q-1)*Tr; % текущее время
       for k=1:K
           RT1(k,q)=sqrt((x(k)+Vx(k)*t-(Xrls(1)+Vxrls*t))^2+...
                         (y(k)+Vy(k)*t-(Yrls(1)+Vyrls*t))^2+...
                         (z(k)+Vz(k)*t-(Zrls(1)+Vzrls*t))^2);
       end  
   end
   Akcomp=sqrt(df*T0*SigBt); U01comp=zeros(N,Q); 
   for q=1:Q
       for k=1:K
           n0=round((2*RT1(k,q)/c-t2+T0)*Fs);  
           n2=n0+gamma; if n2>N  n2=N; end
           n1=n0-gamma; if n1<1  n1=1; end
           for n=n1:n2 % добавляем отсчеты сигнала
               t=t2+n/Fs-(2*RT1(k,q)/c+T0);
               U01comp(n,q)=U01comp(n,q)+Akcomp(k)*sinc(df*t)*exp(j*(4*pi*RT1(k,q)/lamda+FiBT(k))); % *exp(j*pi*Fs/Ks*t);
           end
       end    
   end
end

% расчет для передающего и двух приемных каналов 
if FlagInter==1
    RT1s=zeros(K,Q+1); RT2s=zeros(K,Q+1);
    for q=1:Q+1
        t=(q-1)*Tr; % текущее время
        for k=1:K
            % дальность между передатчиком и БТ
            r00=sqrt((x(k)+Vx(k)*t-(Xrls(1)+Vxrls*t))^2+...
                     (y(k)+Vy(k)*t-(Yrls(1)+Vyrls*t))^2+...
                     (z(k)+Vz(k)*t-(Zrls(1)+Vzrls*t))^2);
            % первая суммарная дальность
            RT1s(k,q)=sqrt((x(k)+Vx(k)*t-(Xrls(2)+Vxrls*t))^2+...
                           (y(k)+Vy(k)*t-(Yrls(2)+Vyrls*t))^2+...
                           (z(k)+Vz(k)*t-(Zrls(2)+Vzrls*t))^2)+r00;
            % вторая суммарная дальность
            RT2s(k,q)=sqrt((x(k)+Vx(k)*t-(Xrls(3)+Vxrls*t))^2+...
                           (y(k)+Vy(k)*t-(Yrls(3)+Vyrls*t))^2+...
                           (z(k)+Vz(k)*t-(Zrls(3)+Vzrls*t))^2)+r00;
        end
    end
    % генерация сигнала
    Akcomp=sqrt(df*T0*Ps); 
    U01comp=zeros(N,Q); U02comp=zeros(N,Q);
    for q=1:Q
        for k=1:K
            % первый приемный канала
            n0=round((RT1s(k,q)/c-t2+T0)*Fs);  
            n2=n0+gamma; if n2>N  n2=N; end % максимальное значение m
            n1=n0-gamma; if n1<1  n1=0; end % минимальное значение m
            for n=n1:n2 % добавляем отсчеты сигнала
                t=t2+n/Fs-(RT1s(k,q)/c+T0);
                U01comp(n,q)=U01comp(n,q)+Akcomp(k)*sinc(df*t)*exp(j*(2*pi*RT1s(k,q)/lamda+FiBT(k)))*exp(j*pi*Fs/Ks*t);
            end
            % второй приемный канала
            n0=round((RT2s(k,q)/c-t2+T0)*Fs);  
            n2=n0+gamma; if n2>N  n2=N; end % максимальное значение m
            n1=n0-gamma; if n1<1  n1=0; end % минимальное значение m
            for n=n1:n2 % добавляем отсчеты сигнала
                t=t2+n/Fs-(RT2s(k,q)/c+T0);
                U02comp(n,q)=U02comp(n,q)+Akcomp(k)*sinc(df*t)*exp(j*(2*pi*RT2s(k,q)/lamda+FiBT(k)))*exp(j*pi*Fs/Ks*t);
            end
        end % по всем БТ
    end % по всем периодам повторения
end

% вычисляем спектр сжатого сигнала приемном канале
Y001=fft(U01comp);      % БПФ сжатого сигнала в 1-ом пк
if FlagInter==1 
    Y002=fft(U02comp);  % БПФ сжатого сигнала во 2-ом пк
end

% восстанавливаем спектр сигнала на входе согласованного фильтра 
[Gh0max, nmax]=max(abs(Gh0(1:N/4)));
for q=1:Q
    for n=1:N
        if n>=round(0.75*nmax) && n<=N/Ks-round(0.75*nmax)
            Y001(n,q)=Y001(n,q)/Gh0(n); 
            if FlagInter==1 
                Y002(n,q)=Y002(n,q)/Gh0(n);
            end    
        else
            Y001(n,q)=0;
            if FlagInter==1 
                Y002(n,q)=0;
            end    
        end    
    end
end

% восстановленный траекторный  сигнал
Y01v=ifft(Y001); if FlagInter==1 Y02v=ifft(Y002); end
% добавляем шум
SigNoise=sqrt(Pnoise/2); % СКО квадратурных компонентов шума
Y01v=Y01v+SigNoise*(randn(N,Q)+j*randn(N,Q));
Y01=fft(Y01v);    % спектр восстановленного сигнала с шумом
if FlagInter==1 % если интерферометрический режим
   Y02v=Y02v+SigNoise*(randn(N,Q)+j*randn(N,Q));
   Y02=fft(Y02v); % спектр восстановленного сигнала с шумом
end

% квантование траекторного сигнала в соответствии с разрядностью АЦП
Y01v=floor(Y01v/dacp);
Y02v=floor(Y02v/dacp);

t=clock; disp(['Сигнал рассчитан ',num2str(t(1,4)),'.',num2str(t(1,5)),'.',num2str(t(1,6)),'.' ]); 