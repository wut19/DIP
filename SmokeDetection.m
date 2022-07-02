function SmokeDetection(m)
    %根据opencv源代码改编 输入视频 输出一连串的黑白图像帧
    %   各个模块的置信度
%     Kc = 0.25;  %颜色判据
%     Kd = 0.25;  %扩散性判据
%     Ks = 0.25;  %不规则形判据
%     Ke = 0.25;  %高频能量判据

    %一些参数
    %颜色判断参数
%     R1 = 0.314;
%     R2 = 0.3369;
%     G1 = 0.3190;
%     G2 = 0.3374;
%     Tc = 10;
    % 扩散特征参数
    delta1 = 20;
    delta2 = 20;
%     v1 = 1;
%     v2 = 1;
    %


    %-------混合高斯背景建模 参数 -----------------
    gauss_n = 3;     %每个像素点高斯背景模型数量
    a   = 0.1;     %学习速率   alpha
    vt  = 2.5^2;     %方差阈值   2.5*2.5倍的方差VarThreshold 
    bgr = 0.7;       %背景比率   BackgroundRatio 
    w0  = 0.05;      %初始权值   weight
    var0= 10^2;      %初始方差   variance 
    path = sprintf('smoke_video/smoke%02d.mp4',m);
    v = VideoReader(path);    %读取视频
    % v1 = VideoWriter('Csmoke02_gbe','MPEG-4');
    % open(v1);

    %-------混合高斯背景建模 读取视频参数----------------
    f_n  =   v.NumFrames;        %帧数 frame_num 
    f_Gray    =   rgb2gray(read(v,1));     %读取第一帧灰度图像  
    f_Gray = imresize(f_Gray,[240,320]);    %调整分辨率
%     f_RGB = read(v,1);                      
%     f_RGB = imresize(f_RGB,[240,320]);      %调整分辨率
    height = 240;   %v.Height;                %获取图像的高度
    width  = 320;   %v.Width;                 %获取图像的宽度
    S = zeros(1,f_n);         %面积
    D = zeros(1,f_n);         %高频能量衰减度
    D1 = zeros(1,f_n);
    Cm = zeros(1,f_n);         %符合特征颜色区域大小
    dS = zeros(1,f_n);        %面积变化率
    std = zeros(1,f_n);       %不规则度
    Rl = zeros(1,f_n);         %相关性
    Rl1 = zeros(1,f_n); 
    Rl2 = zeros(1,f_n); 
    Rl3 = zeros(1,f_n); 

    %--------初始化高斯背景模型 共有height*width*gauss_n*3个数值-
    %每一个像素对应 gauss_n 个高斯背景模型  每个模型有三个参数[权值 均值 方差]
    g_b = zeros(height,width,gauss_n,3); 
    for h = 1:height
        for w = 1:width           %像素遍历
            g_b(h,w,1,1) = 1;     %第一个模型初始权值为1
            g_b(h,w,1,2) = double(f_Gray(h,w));%第一个模型初始均值为第一帧灰度图像素点的值
            g_b(h,w,1,3) = 9;    %初始方差
        end
    end %此方式初始化容易将第一帧内的运动物体也当成背景 最好使用前n个帧训练模型 or 一开始的学习率很高

    %---------进行匹配 更新模型---------------
    %帧遍历
%     fc_last = zeros(240,320);
    f_Graylast = zeros(240,320);
    for n=2:f_n  
        f_Gray = rgb2gray(read(v,n));       %读取下一帧
        f_Gray = imresize(f_Gray,[240,320]);    %调整分辨率
        f_RGB = read(v,1);
        f_RGB = imresize(f_RGB,[240,320]);      %调整分辨率
        %像素遍历
        f_GrayBin = zeros(height,width);
        for h = 1:height
            for w = 1:width    
                khit = 0;   %匹配的模型序号 默认与第一个模型匹配
                bg_n = 0;   %描述背景的高斯模型数量
                %高斯模型遍历
                for k = 1:gauss_n
                    ww   = g_b(h,w,k,1);        %模型权值
                    if(ww == 0)                 %权值为0 则模型为空 跳过
                        continue;
                    end
                    mean1 = g_b(h,w,k,2);        %模型均值
                    var  = g_b(h,w,k,3);        %模型方差
                    diff = double(f_Gray(h,w))-mean1; %像素点与模型均值的差 
                    d2 = diff^2;          %差的平方
                    %与此模型匹配成功
                    if(d2 < vt*var)  
                        g_b(h,w,k,1) =  ww + a * (1 - ww);     %增加权值  
                        g_b(h,w,k,2) =  mean1 + a * diff;       %更新均值
                        g_b(h,w,k,3) =  var + a * (d2 - var);  %更新方差
                        khit = k;  %记录匹配的模型序号
                        %模型排序 从后向前冒泡
                        for kk = k:-1:2
                            ww1 = g_b(h,w,kk,1);%权值
                            var1= g_b(h,w,kk,3);%方差 
                            ww  = g_b(h,w,kk-1,1);%权值
                            var = g_b(h,w,kk-1,3);%方差 
                            %大于前一个 则交换
                            if(ww1/sqrt(var1) > ww/sqrt(var))
                                tmp = g_b(h,w,kk,:);
                                g_b(h,w,kk,:) = g_b(h,w,kk-1,:);
                                g_b(h,w,kk-1,:) = tmp;
                                khit = khit - 1; %匹配的模型序号更新
                            end
                        end
                        break;
                    end
                end
                %全部匹配失败  新建立模型覆盖权值为0 or 最后一个模型
                if(khit == 0)
                    for k = 2:gauss_n
                        if(g_b(h,w,k,1) == 0 || k == gauss_n)
                            g_b(h,w,k,1) = w0;
                            g_b(h,w,k,2) = double(f_Gray(h,w));
                            g_b(h,w,k,3) = var0;
                            break;
                        end
                    end
                    khit = k;      %匹配的模型序号变更
                end
                %权值归一化 保证权值和为1
                wsum = sum( g_b(h,w,:,1) );
                bt = 0;
                for k = 1:gauss_n
                    %%%
                    g_b(h,w,k,1) = g_b(h,w,k,1)/wsum;
                    bt = bt + g_b(h,w,k,1);
                    %前bg_n个模型的权值和 大于背景比率 则前gb_n个模型来描述背景
                    if( bt > bgr && bg_n ==0)
                        bg_n = k;
                    end
                end

                %二值化 
                if(khit > bg_n)  %匹配的模型 不是前gb_n描述背景的模型
                    f_GrayBin(h,w) = 255;
                else             %匹配的模型 属于用来描述背景的模型
                    f_GrayBin(h,w) = 0;
                end
            end
        end
        % 开闭运算去噪
        f_GrayBin = imbinarize(f_GrayBin);
        f_GrayBin = bwmorph(f_GrayBin,'close');
        f_GrayBin = bwmorph(f_GrayBin,'open');

        %总判据清零
%         K = 0;
        % RGB判据
        %获取三个通道的像素
        fr = f_RGB(:,:,1);
        fg = f_RGB(:,:,2);
        fb = f_RGB(:,:,3);
        fr_move = double(fr).*f_GrayBin;
        fg_move = double(fg).*f_GrayBin;
        fb_move = double(fb).*f_GrayBin;

        %进行判断
%         frBin = fr_move>R1*255 & fr_move<R2*255;
%         fgBin = fg_move>G1*255 & fr_move<G2*255;
%         frgBin = fr_move<fg_move;
%         fc = frBin & fgBin & frgBin;  %得到颜色判据判断后的区域
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

        %扩散性判据
        S(n) = sum(fc,'all');    %面积
        if n-delta1-delta2>1
           vs=mean((S(n-delta1:n)-S(n-delta1-delta2:n-delta2))./S(n-delta1-delta2:n-delta2));
%            if vs>v1 && vs<v2
%                K = K + Kd; 
%            end
           dS(n) = vs;
        end

        %不规则判据
        %首先构建周长计算矩阵
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
        C = C(fc);   %获取ROI像素点上的结果
        SEP = sum(C);       %将结果求和得到周长
        STD = SEP/S(n); %得到不规则度
        std(n) = STD;

        %高频能量判据
        f_c1 = imresize(fc,[120,160]);
        %先进行小波变换得到系数
        [~,cH,cV,cD]=dwt2(f_Gray,'haar');%使用haar小波
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

        % 相关
        [~,cH2,cV2,cD2] = dwt2(f_Graylast,'haar');    %上一张灰度图片的小波
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
    %     %输出
    %     clc;
        fprintf('进度：%d / %d \n',n,f_n); 
    %     writeVideo(v1,fc*1.0);
    %     imwrite(f,strcat('output/dst_',num2str(n),'.jpg'));
    end
    disp('OK!');
    path = sprintf('data/smoke%02d.mat',m);
    save(path,'dS','std','D','Rl');
end
% close(v1);
%%-----小结------
%1.视频先用软件进行预处理(裁剪，缩小) 方便测试程序 
%2.图片像素点类型为uint8 改为double参与运算才可得负数   (class 查看变量类型)
%3.打印某帧参数到txt       windows \r\n换行
%  1）fid=fopen(strcat(num2str(n),'.txt'),'w');    %创建日志
%  2）fprintf(fid,'(%d,%d)---%d\r\n',h,w,f(h,w));  %写入日志
%  3）fclose(fid);                                 %关闭
%