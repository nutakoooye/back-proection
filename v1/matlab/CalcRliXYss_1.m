% Ïîñòðîåíèå ÐËÈ â êîîðäèíàòàõ (x,y) äëÿ óâåëè÷åííîé ÷àñòîòû äèñêðåòèçàöèè
% Âõîä: ìàññèâû Y001,Y002 ðàçìåðíîñòüþ N,Q ñî ÑÏÅÊÒÐÀÌÈ òðàåêòîðíûõ
% ñèãíàëîâ; N - ÷èñëî îòñ÷åòîâ ïî äàëüíîñòè; Q - ÷èñëî ïåðèîäîâ ïîâòîðåíèÿ 
% Ïðè ÂÏÎ ïðîâîäèòñÿ óâåëè÷åíèå ÷àñòîòû äèñêðåòèçàöèè òðàåêòîðíîãî ñèãíàëà çà ñ÷åò
% äîïîëíåíèÿ ñïåêòðà íóëÿìè; âû÷èñëÿåòñÿ ÈÕ è Ê×Õ ñîãëàñîâàííîãî ôèëüòðà ÂÏÎ ñ 
% ó÷åòîì îêîííîé ôóíêöèè ïî äàëüíîñòè, ïåðåìíîæåíèå ñïåêòðà è Ê×Õ,
%  îáðàòíîå ÄÏÔ.
% Äàëåå ïðîâîäèòñÿ ñèíòåç äëÿ óâåëè÷åííîé ÷àñòîòû äèñêðåòèçàöèè, äëÿ ÷åãî 
% ñ çàäàííîé äèñêðåòíîñòüþ ïðîñìàòðèâàþèñÿ âñå òî÷êè çàäàííîé çîíû ñèíòåçèðîâàíèÿ. 
% Äëÿ êàæäîé òî÷êè âûáèðàåòñÿ ñèììåòðè÷íûé èíòåðâàë ñèíòåçèðîâàíèÿ îòíîñèòåëüíî òðàâåðñà
% Äëÿ êàæäîãî ïîëîæåíèÿ ÐÑÀ â çîíå ñèíòåçèðîâàíèÿ ïðîâîäèòñÿ
% âû÷èñëåíèå äàëüíîñòè äî òî÷êè, èíäåêñ ñæàòîãî ñèãíàëà è ïðîâîäèòñÿ 
% ñóììèðîâàíèå ýòèõ ñèãíàëîâ ñ êîìïåíñàöèåé èçìåíåíèÿ ôàçû. 
% Ïðèåìóùåñòâà: íåò íåîáõîäèìîñòè â ïîñëåäóþùèõ ïðåîáðàçîâàíèÿõ èç 
% êîîðäèíàò "íàêëîííàÿ äàëüíîñòü-ïîïåðå÷íàÿ äàëüíîñòü" â êîîðäèíàòû (x,y)  

%%%%%%%%%%%%%%%%%%%%%%% ÏÀÐÀÌÅÒÐÛ ÎÁÐÀÁÎÒÊÈ %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% çàäàåì çîíó ñèíòåçèðîâàíèÿ
dxsint=3.0; % äèñêðåòíîñòü ïî Ox
dysint=3;   % äèñêðåòíîñòü ïî Oy
% ðàçìåðû è ÷èñëî òî÷åê ïî Ox
X1sint=min(x)-200; X2sint=max(x)+200; Nxsint=round((X2sint-X1sint)/dxsint); 
% ðàçìåðû è ÷èñëî òî÷åê ïî Oy
Y1sint=-1000;      Y2sint=1000;      Nysint=round((Y2sint-Y1sint)/dysint); 
% îêîííûå ôóíêöèè:
% 1 - ïðÿìîóãîëüíîå îêíî
% 2 - êîñèíóñ íà ïúåäåñòàëå, ïðè delta=0.54 - Õýììèíãà (-42.7 äÁ)
% 3 - êîñèíóñ êâàäðàò íà ïúåäåñòàëå
% 4 - Õýììèíãà (-42.7 äÁ)
% 5 - Õýííèíãà â òðåòüå ñòåïåíè íà ïúåäåñòàëå (-39.3 äÁ)
% 6 - Õýííèíãà â ÷åòâåðòîé ñòåïåíè íà ïúåäåñòàëå (-46.7 äÁ)
% 7,8,9 - Êàéçåðà-Áåññåëÿ äëÿ  alfa=2.7; 3.1; 3.5 (-62.5; -72þ1; -81.8 äÁ)
% 10 - Áëåêìàíà-Õåððèñà (-92 äÁ)
TypeWinDn=1; % òèï îêîííîé ôóíêöèè ïðè ñæàòèè ïî íàêëîííîé äàëüíîñòè 
TypeWinDp=1; % òèï îêîííîé ôóíêöèè ïðè ñæàòèè ïî ïîïåðå÷íîé äàëüíîñòè  

% äîáàâëÿåì øóì (óáðàíî)
% SigNoise=0/sqrt(2); % ÑÊÎ êâàäðàòóðíûõ êîìïîíåíòîâ øóìà
% Y01=Y001+SigNoise*(randn(N,Q)+j*randn(N,Q));
% if FlagInter==1 % åñëè èíòåðôåðîìåòðè÷åñêèé ðåæèì
%    Y02=Y002+SigNoise*(randn(N,Q)+j*randn(N,Q));
% end

%%%%%%%%%%%%%%%%%%%   ÂÏÎ ñ ïåðåäèñêðåòèçàöèåé  %%%%%%%%%%%%%%%%%%%%%%%%%%
% ïåðåäèñêðåòèçàöèÿ ïî íàêëîííîé äàëüíîñòè - äîïîëíåíèå ñïåêòðà íóëÿìè
Y01ss=zeros(Kss*N,Q); if FlagInter==1 Y02ss=zeros(Kss*N,Q); end
% íîâûé âàðèàíò ïåðåäèñêðåòèçàöèè - ïðîñòî äîáàâëÿåì íóëè
for q=1:Q
    for n=1:N
        Y01ss(n,q)=Y01(n,q); if FlagInter==1 Y02ss(n,q)=Y02(n,q); end
    end
end    

% âû÷èñëåíèå Ê×Õ ôèëüòðà ñ óâåëè÷åííûì ÷èñëîì îòñ÷åòîâ ñ ó÷åòîì îêîííîé ôóíêöèè 
h0ss=zeros(N*Kss,1);
for n=1:T0*Fs*Kss+1;
    h0ss(n,1)=conj(ChirpSig(T0-(n-1)/(Fs*Kss),T0,b))*...
              1; 
end
Gh0ss=fft(h0ss); Gh0ss=Gh0ss/sqrt(N*Kss); 

% ñæàòèå ïî äàëüíîñòè ïî 1-ìó ïðèåìíîìó êàíàëó
Goutss=zeros(N*Kss,Q);
for q=1:Q
    Goutss(:,q)=Y01ss(:,q).*Gh0ss(:,1);
end
Uout01ss=ifft(Goutss)*sqrt(N*Kss);

% ñæàòèå ïî äàëüíîñòè ïî 2-ìó ïðèåìíîìó êàíàëó
if FlagInter==1
   Goutss=zeros(N*Kss,Q);
   for q=1:Q
       Goutss(:,q)=Y02ss(:,q).*Gh0ss(:,1);
   end
Uout02ss=ifft(Goutss)*sqrt(N*Kss);
end

%%%%%%%% ñæàòèå â êîîðäèíàòàõ (x,y) ñïîñîáîì Backprojection  %%%%%%%%%%%%%%%
% îòñ÷åòû íàêîïëåííîãî êîìïëåêñíîãî ÐËÈ-1
Zxy1=zeros(Nxsint,Nysint); 
if FlagInter==1 Zxy2=zeros(Nxsint,Nysint); end 
% ïîäãîòîâêà çíà÷åíèé îêîííîé ôóíêöèè
Nr=round(Tsint/Tr); % ÷èñëî ïåðèîäîâ ïîâòîðåíèÿ íà èíòåðâàëå ñèíòåçèðîâàíèÿ
for q=1:Nr+1
    1;
end    

% îñíîâíîé ðàñ÷åòíûé öèêë ÄËß ÌÀÊÑÈÌÀËÜÍÎÃÎ ÐÀÑÏÀÐÀËËÅËÈÂÀÍÈß
for nx=1:Nxsint
    for ny=1:Nysint
        % êîîðäèíàòû òåêóùåé òî÷êè íàáëþäåíèÿ
        xt=X1sint+(nx-1)*dxsint; yt=Y1sint+(ny-1)*dysint; zt=0;
        % îïðåäåëÿåì èíòåðâàë èíäåêñîâ îòñ÷åòîâ äëÿ ñóììèðîâàíèÿ îòñ÷åòîâ èñõîäÿ èç
        % íàõîæäåíèÿ íà òðàâåðñå äëÿ ïåðåäàþùåãî êàíàëà
        q0=round((yt-Yrls(1))/(Vyrls*Tr)); % èíäåêñ íîìåðà ïåðèîäà ïîâòîðåíèÿ äëÿ òðàâåðñà
        q1=q0-round(Tsint/2/Tr); % íà÷àëüíûé èíäåêñ ñóììèðîâàíèÿ
        q2=q0+round(Tsint/2/Tr); % êîíå÷íûé èíäåêñ ñóììèðîâàíèÿ
        d0=sqrt((Xrls(1)-xt)^2+(Zrls(1)-zt)^2); % äàëüíîñòü íà òðàâåðñå
        ar=Vrls^2/d0;  % ðàäèàëüíîå óñêîðåíèå äëÿ êîìïåíñàöèè ÌÄ è Ì×
        % íåñêîìïåíñèðîâàííûå ñêîðîñòè äëÿ ïðèåìíûõ êàíàëîâ
        Vr1=L/2/d0*Vrls; Vr2=-L/2/d0*Vrls;  
        % íåïîñðåäñòâåííî ñóììèðîâàíèå - ÊÍ ñ êîìïåíñàöèåé ÌÄ è Ì×      
        if FlagInter==0 % åñëè îäèí êàíàë ðàáîòàåò
           sum1=0;  
           for q=q1:q2 % ñóììèðîâàíèå èìïóëüñîâ
                % äàëüíîñòü ìåæäó òî÷êîé ñèíòåçèðîâàíèÿ è ÐËÑ â q-îì
                % ïåðèîäå ïîâòîðåíèÿ
                d=sqrt((xt-(Xrls(1)+Vxrls*((q-1)*Tr)))^2+...
                       (yt-(Yrls(1)+Vyrls*(q-1)*Tr))^2+...
                       (zt-(Zrls(1)+Vzrls*(q-1)*Tr))^2);
                % äðîáíûé íîìåð îòñ÷åòà, ãäå íàõîäèòñÿ ñèãíàë, ïî áûñòðîìó âðåìåíè
                ndr=(2*d/c-t2+T0)*Kss*Fs;  
                % öåëàÿ è äðîáíàÿ ÷àñòü èíäåêñà
                n=floor(ndr)+1; drob=mod(ndr,1);
                % ëèíåéíàÿ èíòåðïîëÿöèÿ îòñ÷åòîâ
                ut=Uout01ss(n,q)*(1-drob)+Uout01ss(n+1,q)*drob;
                % ñóììèðîâàíèå ñ ó÷åòîì ïîâîðîòà ôàçû è âçâåøèâàíèÿ
                sum1=sum1+ut*WinSampl(q-q1+1)*exp(-j*4*pi/lamda*ar/2*((q-q0)*Tr)^2);
           end
           Zxy1(nx,ny)=sum1;
        end
        if FlagInter==1 % åñëè èíòåðôåðîìåòðè÷åñêèé ðåæèì ñ äâóìÿ ïðèåìíûìè êàíàëàìè
           sum1=0; sum2=0;
           for q=q1:q2
                d=sqrt((xt-(Xrls(1)+Vxrls*((q-1)*Tr)))^2+...
                       (yt-(Yrls(1)+Vyrls*(q-1)*Tr))^2+...
                       (zt-(Zrls(1)+Vzrls*(q-1)*Tr))^2);
                ndr=(2*d/c-t2+T0)*Kss*Fs; % äðîáíûé íîìåð îòñ÷åòà ïî áûñòðîìó âðåìåíè 
                n=floor(ndr)+1; drob=mod(ndr,1); 
                ut=Uout01ss(n,q)*(1-drob)+Uout01ss(n+1,q)*drob;
                % ñóììèðóåì ñ ó÷åòîì ñäâèãà ÐËÈ ïî ñêîðîñòè
                sum1=sum1+ut*1*exp(-j*4*pi/lamda*ar/2*((q-q0)*Tr)^2)*exp(j*2*pi*Vr1/lamda*(q-q0)*Tr);
                ut=Uout02ss(n,q)*(1-drob)+Uout02ss(n+1,q)*drob;
                sum2=sum2+ut*1*exp(-j*4*pi/lamda*ar/2*((q-q0)*Tr)^2)*exp(j*2*pi*Vr2/lamda*(q-q0)*Tr);
           end
           Zxy1(nx,ny)=sum1; Zxy2(nx,ny)=sum2;
        end
    end
end    

% îòîáðàæåíèå
figure; surfl(abs(Zxy1).^2); view(45,30);  
shading flat; shading interp; colormap parula;
xlabel('Oy'); ylabel('Ox'); title('ÐËÈ ïî ïåðâîìó êàíàëó');
if FlagInter==1
   figure; surfl(abs(Zxy2).^2); view(45,30);  
   shading flat; shading interp; colormap parula;
   xlabel('Oy'); ylabel('Ox'); title('ÐËÈ ïî âòîðîìó êàíàëó');
   figure; surfl(abs(Zxy1-Zxy2).^2); view(45,30);  
   shading flat; shading interp; colormap parula;
   xlabel('Oy'); ylabel('Ox'); title('ðàçíîñòíîå ÐËÈ');
end

% âûâîä ìàêñèìóìîâ è ðàçíîñòè ôàç
[Zxy1max,nqmax]=(max(abs(Zxy1),[],'all','linear'));
nmax1=mod(nqmax,Nxsint); qmax1=floor(nqmax/Nxsint)+1;
disp(['Zxy1(',num2str(nmax1),',',num2str(qmax1),')=',num2str(Zxy1max),' x=', num2str(X1sint+(nmax1-1)*dxsint),' y=', num2str(Y1sint+(qmax1-1)*dxsint)]);
if FlagInter==1
   [Zxy2max,nqmax]=(max(abs(Zxy2),[],'all','linear'));
   nmax2=mod(nqmax,Nxsint); qmax2=floor(nqmax/Nxsint)+1;
   disp(['Zxy2(',num2str(nmax2),',',num2str(qmax2),')=',num2str(Zxy2max),' x=', num2str(X1sint+(nmax2-1)*dxsint),' y=', num2str(Y1sint+(qmax2-1)*dxsint)]);
   disp(['Ðàçíîñòü ôàç=',num2str(angle(Zxy1(nmax1,qmax1))-angle(Zxy2(nmax2,qmax2)))]);
end