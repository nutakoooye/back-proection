% ���������� ��� ��������� ������� �� ������ ��� � ������� �� 
% ������ ��� ����� ����������� �� � ��
figure
Grli0=(abs(Zxy1).^1);
Grli0=1*Grli0/max(max(Grli0));
Grli0=flipud(Grli0');
imshow(Grli0);
xlabel('��'); ylabel('�y');