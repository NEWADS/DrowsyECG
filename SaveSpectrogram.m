%This Function runs spectrogram on ECG signal, its details can be seen in
%function, and save the grams as .jpg files.
for i = 1:size(ECG0, 1)
    input = ECG0(i,:);
    [S,F,T,P] = spectrogram(input,512,475,512,1000); %对应参数：输入数据，窗宽，重叠点个数，采样频率。
    test=surfc(F,T,(10*log10(P))','edgecolor','none');view(2);axis off;
%     saveFile='saveas(gcf,''C:\\Users\\4 -18\\ECGFeatures_ver39samples\\ECGFeatures_ver39samples\\spectrogram\\ECG0\\%d.bmp'');';
%     bb = sprintf(saveFile,i);
    eval(['saveas(gcf,''C:\\Users\\sidgh\\PycharmProjects\\DrowsyECG\\dataset\\Specgrams-unclipped\\0\\',num2str(i),'.bmp'');']);
end
for i = 1:size(ECG1, 1)
    input = ECG1(i,:);
    [S,F,T,P] = spectrogram(input,512,475,512,1000); %对应参数：输入数据，窗宽，重叠点个数，采样频率。
    test=surfc(F,T,(10*log10(P))','edgecolor','none');view(2);axis off;
%     saveFile='saveas(gcf,''E:\\实验室\\4 -18\\ECGFeatures_ver39samples\\ECGFeatures_ver39samples\\spectrogram\\ECG1\\%d.bmp'');';
%     bb = sprintf(saveFile,i);
    eval(['saveas(gcf,''C:\\Users\\sidgh\\PycharmProjects\\DrowsyECG\\dataset\\Specgrams-unclipped\\1\\',num2str(i),'.bmp'');']);
end
log