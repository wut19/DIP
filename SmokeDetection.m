function SmokeDetection(m)
    %����opencvԴ����ı� ������Ƶ ���һ�����ĺڰ�ͼ��֡
    %   ����ģ������Ŷ�
%     Kc = 0.25;  %��ɫ�о�
%     Kd = 0.25;  %��ɢ���о�
%     Ks = 0.25;  %���������о�
%     Ke = 0.25;  %��Ƶ�����о�

    %һЩ����
    %��ɫ�жϲ���
%     R1 = 0.314;
%     R2 = 0.3369;
%     G1 = 0.3190;
%     G2 = 0.3374;
%     Tc = 10;
    % ��ɢ��������
    delta1 = 20;
    delta2 = 20;
%     v1 = 1;
%     v2 = 1;
    %


    %-------��ϸ�˹������ģ ���� -----------------
    gauss_n = 3;     %ÿ�����ص��˹����ģ������
    a   = 0.1;     %ѧϰ����   alpha
    vt  = 2.5^2;     %������ֵ   2.5*2.5���ķ���VarThreshold 
    bgr = 0.7;       %��������   BackgroundRatio 
    w0  = 0.05;      %��ʼȨֵ   weight
    var0= 10^2;      %��ʼ����   variance 
    path = sprintf('smoke_video/smoke%02d.mp4',m);
    v = VideoReader(path);    %��ȡ��Ƶ
    % v1 = VideoWriter('Csmoke02_gbe','MPEG-4');
    % open(v1);

    %-------��ϸ�˹������ģ ��ȡ��Ƶ����----------------
    f_n  =   v.NumFrames;        %֡�� frame_num 
    f_Gray    =   rgb2gray(read(v,1));     %��ȡ��һ֡�Ҷ�ͼ��  
    f_Gray = imresize(f_Gray,[240,320]);    %�����ֱ���
%     f_RGB = read(v,1);                      
%     f_RGB = imresize(f_RGB,[240,320]);      %�����ֱ���
    height = 240;   %v.Height;                %��ȡͼ��ĸ߶�
    width  = 320;   %v.Width;                 %��ȡͼ��Ŀ��
    S = zeros(1,f_n);         %���
    D = zeros(1,f_n);         %��Ƶ����˥����
    D1 = zeros(1,f_n);
    Cm = zeros(1,f_n);         %����������ɫ�����С
    dS = zeros(1,f_n);        %����仯��
    std = zeros(1,f_n);       %�������
    Rl = zeros(1,f_n);         %�����
    Rl1 = zeros(1,f_n); 
    Rl2 = zeros(1,f_n); 
    Rl3 = zeros(1,f_n); 

    %--------��ʼ����˹����ģ�� ����height*width*gauss_n*3����ֵ-
    %ÿһ�����ض�Ӧ gauss_n ����˹����ģ��  ÿ��ģ������������[Ȩֵ ��ֵ ����]
    g_b = zeros(height,width,gauss_n,3); 
    for h = 1:height
        for w = 1:width           %���ر���
            g_b(h,w,1,1) = 1;     %��һ��ģ�ͳ�ʼȨֵΪ1
            g_b(h,w,1,2) = double(f_Gray(h,w));%��һ��ģ�ͳ�ʼ��ֵΪ��һ֡�Ҷ�ͼ���ص��ֵ
            g_b(h,w,1,3) = 9;    %��ʼ����
        end
    end %�˷�ʽ��ʼ�����׽���һ֡�ڵ��˶�����Ҳ���ɱ��� ���ʹ��ǰn��֡ѵ��ģ�� or һ��ʼ��ѧϰ�ʺܸ�

    %---------����ƥ�� ����ģ��---------------
    %֡����
%     fc_last = zeros(240,320);
    f_Graylast = zeros(240,320);
    for n=2:f_n  
        f_Gray = rgb2gray(read(v,n));       %��ȡ��һ֡
        f_Gray = imresize(f_Gray,[240,320]);    %�����ֱ���
        f_RGB = read(v,1);
        f_RGB = imresize(f_RGB,[240,320]);      %�����ֱ���
        %���ر���
        f_GrayBin = zeros(height,width);
        for h = 1:height
            for w = 1:width    
                khit = 0;   %ƥ���ģ����� Ĭ�����һ��ģ��ƥ��
                bg_n = 0;   %���������ĸ�˹ģ������
                %��˹ģ�ͱ���
                for k = 1:gauss_n
                    ww   = g_b(h,w,k,1);        %ģ��Ȩֵ
                    if(ww == 0)                 %ȨֵΪ0 ��ģ��Ϊ�� ����
                        continue;
                    end
                    mean1 = g_b(h,w,k,2);        %ģ�;�ֵ
                    var  = g_b(h,w,k,3);        %ģ�ͷ���
                    diff = double(f_Gray(h,w))-mean1; %���ص���ģ�;�ֵ�Ĳ� 
                    d2 = diff^2;          %���ƽ��
                    %���ģ��ƥ��ɹ�
                    if(d2 < vt*var)  
                        g_b(h,w,k,1) =  ww + a * (1 - ww);     %����Ȩֵ  
                        g_b(h,w,k,2) =  mean1 + a * diff;       %���¾�ֵ
                        g_b(h,w,k,3) =  var + a * (d2 - var);  %���·���
                        khit = k;  %��¼ƥ���ģ�����
                        %ģ������ �Ӻ���ǰð��
                        for kk = k:-1:2
                            ww1 = g_b(h,w,kk,1);%Ȩֵ
                            var1= g_b(h,w,kk,3);%���� 
                            ww  = g_b(h,w,kk-1,1);%Ȩֵ
                            var = g_b(h,w,kk-1,3);%���� 
                            %����ǰһ�� �򽻻�
                            if(ww1/sqrt(var1) > ww/sqrt(var))
                                tmp = g_b(h,w,kk,:);
                                g_b(h,w,kk,:) = g_b(h,w,kk-1,:);
                                g_b(h,w,kk-1,:) = tmp;
                                khit = khit - 1; %ƥ���ģ����Ÿ���
                            end
                        end
                        break;
                    end
                end
                %ȫ��ƥ��ʧ��  �½���ģ�͸���ȨֵΪ0 or ���һ��ģ��
                if(khit == 0)
                    for k = 2:gauss_n
                        if(g_b(h,w,k,1) == 0 || k == gauss_n)
                            g_b(h,w,k,1) = w0;
                            g_b(h,w,k,2) = double(f_Gray(h,w));
                            g_b(h,w,k,3) = var0;
                            break;
                        end
                    end
                    khit = k;      %ƥ���ģ����ű��
                end
                %Ȩֵ��һ�� ��֤Ȩֵ��Ϊ1
                wsum = sum( g_b(h,w,:,1) );
                bt = 0;
                for k = 1:gauss_n
                    %%%
                    g_b(h,w,k,1) = g_b(h,w,k,1)/wsum;
                    bt = bt + g_b(h,w,k,1);
                    %ǰbg_n��ģ�͵�Ȩֵ�� ���ڱ������� ��ǰgb_n��ģ������������
                    if( bt > bgr && bg_n ==0)
                        bg_n = k;
                    end
                end

                %��ֵ�� 
                if(khit > bg_n)  %ƥ���ģ�� ����ǰgb_n����������ģ��
                    f_GrayBin(h,w) = 255;
                else             %ƥ���ģ�� ������������������ģ��
                    f_GrayBin(h,w) = 0;
                end
            end
        end
        % ��������ȥ��
        f_GrayBin = imbinarize(f_GrayBin);
        f_GrayBin = bwmorph(f_GrayBin,'close');
        f_GrayBin = bwmorph(f_GrayBin,'open');

        %���о�����
%         K = 0;
        % RGB�о�
        %��ȡ����ͨ��������
        fr = f_RGB(:,:,1);
        fg = f_RGB(:,:,2);
        fb = f_RGB(:,:,3);
        fr_move = double(fr).*f_GrayBin;
        fg_move = double(fg).*f_GrayBin;
        fb_move = double(fb).*f_GrayBin;

        %�����ж�
%         frBin = fr_move>R1*255 & fr_move<R2*255;
%         fgBin = fg_move>G1*255 & fr_move<G2*255;
%         frgBin = fr_move<fg_move;
%         fc = frBin & fgBin & frgBin;  %�õ���ɫ�о��жϺ������
        frgBin = fr_move > fg_move - 20 & fr_move < fg_move + 20;
        frbBin = fr_move > fb_move - 20 & fr_move < fb_move + 20;
        fgbBin = fg_move > fb_move - 20 & fg_move < fb_move + 20;
        I = (fr_move+fg_move+fb_move)/3;
        fIBin = (I<220 & I>150 )|(I<150&I>80);
        fc  = frgBin & frbBin & fgbBin & fIBin ;



        Cm(n) = sum(fc,'all');
%         if sum(fc,'all')>Tc
%             K = K + Kc;
%         end

        %��ɢ���о�
        S(n) = sum(fc,'all');    %���
        if n-delta1-delta2>1
           vs=mean((S(n-delta1:n)-S(n-delta1-delta2:n-delta2))./S(n-delta1-delta2:n-delta2));
%            if vs>v1 && vs<v2
%                K = K + Kd; 
%            end
           dS(n) = vs;
        end

        %�������о�
        %���ȹ����ܳ��������
        C = zeros(height+2,width+2);
        C(2:height+1,2:width+1) = 4*fc;
        for i=-1:2:1
            C1 = zeros(height+2,width+2);
            C1(2+i:height+1+i,2:width+1) = fc; 
            C = C-C1;
        end
        for i=-1:2:1
            C1 = zeros(height+2,width+2);
            C1(2:height+1,2+i:width+1+i) = fc; 
            C = C-C1;
        end   
        C = C(2:height+1,2:width+1);
        C = C(fc);   %��ȡROI���ص��ϵĽ��
        SEP = sum(C);       %�������͵õ��ܳ�
        STD = SEP/S(n); %�õ��������
        std(n) = STD;

        %��Ƶ�����о�
        f_c1 = imresize(fc,[120,160]);
        %�Ƚ���С���任�õ�ϵ��
        [~,cH,cV,cD]=dwt2(f_Gray,'haar');%ʹ��haarС��
        E_img = mean((cH.*f_c1).^2+(cV.*f_c1).^2+(cD.*f_c1).^2,'all');
        bg = mean(g_b(:,:,:,2),3);
        [~,cH1,cV1,cD1]=dwt2(bg,'haar');
        E_bg = mean((cH1.*f_c1).^2+(cV1.*f_c1).^2+(cD1.*f_c1).^2,'all');
        if E_bg == 0
            D1(n) = -10;
        else
            D1(n) = (E_bg - E_img)/E_bg;
        end
        
        if n-delta1>1
            av_d = mean(D1(n-delta1:n));
        else
            av_d = mean(D1(1:n));
        end
        D(n) = av_d; 
        a = 1;

        % ���
        [~,cH2,cV2,cD2] = dwt2(f_Graylast,'haar');    %��һ�ŻҶ�ͼƬ��С��
        if n>2
           if ~any(f_c1,'all')
               Rl(n)=10;
           else
               Rl1(n) = cH(f_c1).'*cH2(f_c1)/(sqrt(sum(cH(f_c1).^2)*sum(cH2(f_c1).^2)));
               Rl2(n) = cV(f_c1).'*cV2(f_c1)/(sqrt(sum(cV(f_c1).^2)*sum(cV2(f_c1).^2)));
               Rl3(n) = cD(f_c1).'*cD2(f_c1)/(sqrt(sum(cD(f_c1).^2)*sum(cD2(f_c1).^2)));
               Rl(n) = (Rl1(n)+Rl2(n)+Rl3(n))/3;
           end
        end

%         fc_last = fc;
        f_Graylast = f_Gray;
    %     %���
    %     clc;
        fprintf('���ȣ�%d / %d \n',n,f_n); 
    %     writeVideo(v1,fc*1.0);
    %     imwrite(f,strcat('output/dst_',num2str(n),'.jpg'));
    end
    disp('OK!');
    path = sprintf('data/smoke%02d.mat',m);
    save(path,'dS','std','D','Rl');
end
% close(v1);
%%-----С��------
%1.��Ƶ�����������Ԥ����(�ü�����С) ������Գ��� 
%2.ͼƬ���ص�����Ϊuint8 ��Ϊdouble��������ſɵø���   (class �鿴��������)
%3.��ӡĳ֡������txt       windows \r\n����
%  1��fid=fopen(strcat(num2str(n),'.txt'),'w');    %������־
%  2��fprintf(fid,'(%d,%d)---%d\r\n',h,w,f(h,w));  %д����־
%  3��fclose(fid);                                 %�ر�
%