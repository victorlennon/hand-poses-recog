#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/nonfree/nonfree.hpp"
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/video/video.hpp>
//#include "opencv/cv.h"
//#include "opencv/cxcore.h"
//#include "opencv/highgui.h"
#include "math.h"
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <complex>
#include <cstdlib>
#include <unistd.h>
#include <stdlib.h>
#include <conio.h>
#include <sstream>
#include <time.h>
#include <windows.h>

using namespace cv;
using namespace std;

typedef MCIERROR WINAPI(*CDROM)(const char*, char*, unsigned, HWND);
CDROM Command;
//contadores para cada gesto
int g0, g1, g2, g3, g4, g5, g6, g7;

bool comparaContorno(const string filename, vector<vector<Point> > contornoFrame) {

    Mat amostra = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

    vector<vector<Point> > contoursAmostra;
    vector<Point> contour;
    findContours(amostra, contoursAmostra, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    if (!contoursAmostra.empty()) {
        contour = contoursAmostra.at(0);
        cout << "ACHOU" << endl;
    }
    //Diferenca entre o contorno e a amostra
    double diff = 99;
    if (!contornoFrame.empty()) {
        //        diff = matchShapes(contour, contornoFrame[0], CV_CONTOURS_MATCH_I1, 0);
        diff = matchShapes(contour, contour, CV_CONTOURS_MATCH_I2, 0);
        cout << "ACHOU NO QUADRO" << endl;
    }

    // ASSOCIA CHAMADA EXTERNA A UM DETERMINADO NUMERO DE CONCAVIDADES(DEFECTS)
    if (diff <= 1) {
        cout << "HIT! Diferenca: " << diff << endl;
        return true;
    }
    cout << "MISS! Diferenca: " << diff << endl;
    return false;
}

void zeraContadores() {
    g0 = 0;
    g1 = 0;
    g2 = 0;
    g3 = 0;
    g4 = 0;
    g5 = 0;
    g6 = 0;
    g7 = 0;
}

void reconheceGesto(int defects, vector<vector<Point> > contornoFrame) {

    switch (defects) {
            //mao fechada ou 1 dedo
        case 0:
            if (comparaContorno("amostra0.jpg", contornoFrame)) {
                g0++;
                if (g0 >= 30) {
                    int iReturn0 = (int) ShellExecute(NULL, "open", "C:\\Windows\\System32\\calc.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 0" << endl;
                }
            } else
                if (comparaContorno("amostra1.jpg", contornoFrame)) {
                g1++;
                if (g1 >= 30) {
                    int iReturn1 = (int) ShellExecute(NULL, "open", "C:\\Windows\\System32\\notepad.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 1" << endl;
                }
            }
            break;

            // 1 dedo ou 2
        case 1:
            if (comparaContorno("amostra1.jpg", contornoFrame)) {
                g1++;
                if (g1 >= 30) {
                    int iReturn1 = (int) ShellExecute(NULL, "open", "C:\\Windows\\System32\\notepad.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 1" << endl;
                }
            } else
                if (comparaContorno("amostra2.jpg", contornoFrame)) {
                g2++;
                if (g2 >= 30) {
                    int iReturn2 = (int) ShellExecute(NULL, "open", "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 2" << endl;
                }
            } else
                if (comparaContorno("amostra6.jpg", contornoFrame)) {
                g6++;
                if (g6 >= 30) {
                    int iReturn2 = (int) ShellExecute(NULL, "open", "C:\\Windows\\System32\\taskmgr.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 6" << endl;
                }
            } else
                if (comparaContorno("amostra7.jpg", contornoFrame)) {
                g7++;
                if (g7 >= 30) {
                    int iReturn2 = (int) ShellExecute(NULL, "open", "C:\\Program Files (x86)\\K-Lite Codec Pack\\Media Player Classic\\mpc-hc.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 7" << endl;
                }
            }
            break;

            // 2 dedos ou 3
        case 2:
            if (comparaContorno("amostra2.jpg", contornoFrame)) {
                g2++;
                if (g2 >= 30) {
                    //                    int iReturn2 = (int) ShellExecute(NULL, "open", "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 2" << endl;
                }
            } else
                if (comparaContorno("amostra3.jpg", contornoFrame)) {
                g3++;
                if (g3 >= 30) {
                    //                    int iReturn3 = (int) ShellExecute(NULL, "open", "C:\\Windows\\System32\\cmd.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 3" << endl;
                }
            } else
                if (comparaContorno("amostra6.jpg", contornoFrame)) {
                g6++;
                if (g6 >= 30) {
                    //                    int iReturn2 = (int) ShellExecute(NULL, "open", "C:\\Windows\\System32\\taskmgr.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 6" << endl;
                }
            } else
                if (comparaContorno("amostra7.jpg", contornoFrame)) {
                g7++;
                if (g7 >= 30) {
                    //                    int iReturn2 = (int) ShellExecute(NULL, "open", "C:\\Program Files (x86)\\K-Lite Codec Pack\\Media Player Classic\\mpc-hc.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 7" << endl;
                }
            }
            break;

            // 3 dedos ou 4
        case 3:
            if (comparaContorno("amostra3.jpg", contornoFrame)) {
                g3++;
                if (g3 >= 30) {
                    //                    int iReturn3 = (int) ShellExecute(NULL, "open", "C:\\Windows\\System32\\cmd.exe", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 3" << endl;
                }
            } else
                if (comparaContorno("amostra4.jpg", contornoFrame)) {
                g4++;
                if (g4 >= 30) {
                    //                    int iReturn4 = (int) ShellExecute(NULL, "open", "C:\\", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 4" << endl;
                }
            }
            break;

            // 4 dedos ou 5
        case 4:
            if (comparaContorno("amostra4.jpg", contornoFrame)) {
                g4++;
                if (g4 >= 30) {
                    int iReturn4 = (int) ShellExecute(NULL, "open", "C:\\", NULL, NULL, SW_SHOWNORMAL);
                    zeraContadores();
                    cout << "GESTO 4" << endl;
                }
            } else
                if (comparaContorno("amostra5.jpg", contornoFrame)) {
                g5++;
                if (g5 >= 30) {
                    Command = (CDROM) GetProcAddress(LoadLibrary("winmm.dll"), "mciSendStringA");
                    Command("Set CDAudio Door Open", NULL, 0, NULL);
                    zeraContadores();
                    cout << "GESTO 5" << endl;
                }
            }
            break;

            // 5 dedos
        case 5:
            if (comparaContorno("amostra5.jpg", contornoFrame)) {
                g5++;
                if (g5 >= 30) {
                    Command = (CDROM) GetProcAddress(LoadLibrary("winmm.dll"), "mciSendStringA");
                    Command("Set CDAudio Door Open", NULL, 0, NULL);
                    zeraContadores();
                    cout << "GESTO 5" << endl;
                }
            }
            break;
        case 6:
            if (comparaContorno("amostra5.jpg", contornoFrame)) {
                g5++;
                if (g5 >= 30) {
                    Command = (CDROM) GetProcAddress(LoadLibrary("winmm.dll"), "mciSendStringA");
                    Command("Set CDAudio Door Open", NULL, 0, NULL);
                    zeraContadores();
                    cout << "GESTO 5" << endl;
                }
            }
            break;
    }
}

void colorReduce(Mat &image, int div = 32) {
    int nl = image.rows; // number of lines
    // total number of elements per line
    int nc = image.cols * image.channels();
    for (int j = 0; j < nl; j++) {
        // get the address of row j
        uchar* data = image.ptr<uchar > (j);
        for (int i = 0; i < nc; i++) {
            // process each pixel ---------------------
            data[i] = data[i] / div * div + div / 2;
            // end of pixel processing ----------------
        } // end of line
    }
}

void subtracao(Mat &frameA, Mat &frameB, Mat &result) {
    Mat temp;
    subtract(frameA, frameB, temp);
    //converte para escala de cinza
    cvtColor(temp, result, CV_BGR2GRAY);
    //    medianBlur(result, result, 7);
}

//filtragem e operacoes morfologicas

void operacoes(Mat &frame, int ksize_mediana, int iter_open, int iter_close) {
    //suavizacao
    medianBlur(frame, frame, ksize_mediana);

    //operacao de abertura
    erode(frame, frame, Mat(), Point(-1, -1), iter_open);
    dilate(frame, frame, Mat(), Point(-1, -1), iter_open);
    //operacao de fechamento
    dilate(frame, frame, Mat(), Point(-1, -1), iter_close);
    erode(frame, frame, Mat(), Point(-1, -1), iter_close);
}

void normalizaRGB(Mat &frame, Mat &result) {

    int step = frame.step;
    int channels = frame.channels();
    float r, g, b;
    uchar* data = frame.data;
    uchar* data_mask = result.data;
    int step_mask = result.step;

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            b = data[i * step + j * channels + 0];
            g = data[i * step + j * channels + 1];
            r = data[i * step + j * channels + 2];
            data_mask[i * step_mask + j * channels + 0] = (255 * b / (b + g + r));
            data_mask[i * step_mask + j * channels + 1] = (255 * g / (b + g + r));
            data_mask[i * step_mask + j * channels + 2] = (255 * r / (b + g + r));
        }
    }
}

void skinRGBDetect(const Mat &frame, Mat &rgb_mask) {
    int step = frame.step;
    int channels = frame.channels();
    int r, g, b;
    uchar* data = frame.data;
    uchar* data_mask = rgb_mask.data;
    int step_mask = rgb_mask.step;

    for (int i = 0; i < frame.rows; i++) {
        for (int j = 0; j < frame.cols; j++) {
            b = data[i * step + j * channels + 0];
            g = data[i * step + j * channels + 1];
            r = data[i * step + j * channels + 2];

            if ((r > 95 && g > 40 && b > 20) &&
                    (max(max(r, g), b) - min(min(r, g), b) > 15) &&
                    (fabs(r - g) > 15) &&
                    ((r > g) && (r > b)))
                data_mask[i * step_mask + j] = 255;
            else
                data_mask[i * step_mask + j] = 0;
        }
    }
}

void trataContorno(const Mat &imagem, Mat &resultado, int thickness = CV_FILLED) {

    vector<vector<Point> > contours, contours2;
    double result = 0, result2 = 0;

    //    medianBlur(imagem, imagem, 7);
    //operacao de abertura
    //    erode(imagem, imagem, Mat(), Point(-1, -1), 8);
    //    dilate(imagem, imagem, Mat(), Point(-1, -1), 8);

    findContours(imagem, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
    //considera apenas contornos com o perimetro dentro da faixa especificada
    int cmin = 400; // minimum contour length
    int cmax = 2800; // maximum contour length    
    vector<vector<Point> >::iterator itc = contours.begin();
    int index = 0;
    while (itc != contours.end()) {
        if (itc->size() < cmin || itc->size() > cmax)
            itc = contours.erase(itc);
        else {
            result = fabs(contourArea(contours.at(index)));
            if (result > result2) {
                result2 = result;
                contours.at(0) = contours.at(index);
            }
            ++itc;
            ++index;
        }
    }
    if (!contours.empty()) {
        contours2.push_back(contours.at(0));
    }
    //remover contorno ruidoso
    medianBlur(resultado, resultado, 3);
    drawContours(resultado, contours2, -1, 255, thickness);
}

void equRGB(Mat &mao, Mat mao_norm) {
    vector<Mat> bgr_planes2;
    split(mao, bgr_planes2);
    equalizeHist(bgr_planes2[0], bgr_planes2[0]);
    equalizeHist(bgr_planes2[1], bgr_planes2[1]);
    equalizeHist(bgr_planes2[2], bgr_planes2[2]);
    merge(bgr_planes2, mao_norm);
}

void histograma(Mat &mao, Mat &mask) {
    // CALCULA HISTOGRAMAS R G B --------------------------------------------------------------------------------------------

    vector<Mat> bgr_planes;
    split(mao, bgr_planes);

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = {0, 256};
    const float* histRange = {range};

    bool uniform = true;
    bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist(&bgr_planes[0], 1, 0, mask, b_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, mask, g_hist, 1, &histSize, &histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, mask, r_hist, 1, &histSize, &histRange, uniform, accumulate);

    // Draw the histograms for B, G and R
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    /// Draw for each channel
    for (int i = 1; i < histSize; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
                Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
                Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
                Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
                Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
                Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
                Scalar(0, 0, 255), 2, 8, 0);
    }
    namedWindow("histograma", CV_WINDOW_AUTOSIZE);
    imshow("histograma", histImage);
}


//alterar range de forma adaptativa (iterativa) até que se tenha um contorno com o tamanho (ou forma) desejado

int main() {


    int c = 0;
    Mat fundoFull;
    Mat maoFull;
    // gambiarra: ao capturar o frame de maoFull, sem motivo aparente o dataBuffer de fundoFull recebe o mesmo valor
    // por isso se faz necessario o artificio de atribuir os dados do fundo para uma estrutura temporaria e depois copiar para fundoFull
    Mat temp;


    //Inicializa frames
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cout << "Captura de video falhou!" << endl;
        return 1;
    }

    //CAPTURA DO FUNDO
    namedWindow("FUNDO", WINDOW_AUTOSIZE);
    cout << "Tecle ESPACO para capturar o plano de fundo" << endl;
    while (c != 32) {
        c = waitKey(0);
    }
    capture.read(temp);
    capture.read(temp);
    temp.copyTo(fundoFull);
    imshow("FUNDO", fundoFull);
    imwrite("C://prints//fundoFull.jpg", fundoFull);

    //CAPTURA DA MAO
    c = 0;
    namedWindow("MAO", WINDOW_AUTOSIZE);
    cout << "Tecle ESPACO com a mao direita aberta posicionada ao lado do seu rosto para calibrar" << endl;
    while (c != 32) {
        c = waitKey();
    }
    //descarta framebuffer ao capturar 2 frames
    capture.read(maoFull);
    capture.read(maoFull);
    imshow("MAO", maoFull);
    imwrite("C://prints//maoFull.jpg", maoFull);

    Size szFull = maoFull.size();
    Size sz(szFull.width / 2, szFull.height);

    //regiao de interesse  = metade esqueda da imagem
    Mat mao;
    Rect roi(0, 0, sz.width, sz.height);
    mao = maoFull(roi);
    rectangle(maoFull, roi, Scalar(0, 0, 0), 2);
    imwrite("C://prints//maoROI.jpg", maoFull);
    imwrite("C://prints//mao.jpg", mao);

    //    imshow("MAO", maoFull);

    Mat fundo;
    Rect roi2(0, 0, sz.width, sz.height);
    fundo = fundoFull(roi2);
    rectangle(fundoFull, roi2, Scalar(0, 0, 0), 2);
    imwrite("C://prints//fundo.jpg", fundo);

    //    imshow("FUNDO", fundoFull);

    //inicializa
    //    Mat frameFull; // current video frame
    Mat rgb_mask(sz, CV_8UC1);
    Mat final_hsv(sz, CV_8UC1);
    Mat final_rgb(sz, CV_8UC1);
    Mat mask(sz, CV_8UC1);
    Mat resultSub(sz, CV_8UC1);
    Mat media_rgb_mask(sz, CV_8UC1);
    Mat bg(sz, CV_8UC3);
    Mat maoEquRGB(sz, CV_8UC3);
    Mat maoCorReduz(sz, CV_8UC3);
    vector<vector<Point> > contours;
    vector<vector<Point> > contours2;

    //IMAGEM ORIGINAL
    //    namedWindow("Frame Original", WINDOW_AUTOSIZE);
    //    imshow("Frame Original", mao);


    //-------------------------------------------------------------
    //OBTENCAO DA MASCARA DE CALIBRACAO

    //verifica brilho da imagem
    Scalar media, desvio;
    Mat brilho(sz, CV_8SC1);
    Mat hsv;
    cvtColor(mao, hsv, CV_BGR2HSV_FULL);
    vector<Mat> planes;
    split(hsv, planes);
    planes[2].copyTo(brilho);
    meanStdDev(brilho, media, desvio);
    //    if (media(0) > 190) {
    //        equRGB(mao, mao); //SE A IMAGEM ESTIVER CLARA
    //        imshow("maoEquRGB", mao);
    //        imwrite("C://prints//maoEquRGB.jpg", mao);
    //    }

    subtracao(fundo, mao, resultSub);
    namedWindow("subtracao", WINDOW_AUTOSIZE);
    imshow("subtracao", resultSub);
    imwrite("C://prints//subtracao.jpg", resultSub);

    //limiarizacao
    threshold(resultSub, resultSub, 10, 255, CV_THRESH_BINARY);
    //TODO enquanto nao achar contorno c/ tamanho razoavel, diminuir o limiar
    //TODO imagens mais claras: limiar maior (50), (testar brilho médio da imagem HSV)

    namedWindow("limiarizacao", WINDOW_AUTOSIZE);
    imshow("limiarizacao", resultSub);
    imwrite("C://prints//limiarizacao.jpg", resultSub);

    operacoes(resultSub, 13, 4, 5);
    namedWindow("operacoes", WINDOW_AUTOSIZE);
    imshow("operacoes", resultSub);
    imwrite("C://prints//operacoes.jpg", resultSub);
    trataContorno(resultSub, mask);

    //    //encontra contornos
    //    findContours(resultSub, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    //
    //    // TODO: tentar fazer isso com a relação entre o tamanho do perimetro e o tamanho da imagem
    //    //considera apenas contornos com o perimetro dentro da faixa especificada
    //    int cmin = 800; // minimum contour length
    //    int cmax = 3000; // maximum contour length
    //    double result, result2;
    //    int index = 0;
    //    vector<vector<Point> >::iterator itc = contours.begin();
    //    while (itc != contours.end()) {
    //        if (itc->size() < cmin || itc->size() > cmax)
    //            itc = contours.erase(itc);
    //        else {
    //            result = fabs(contourArea(contours.at(index)));
    //            if (result > result2) {
    //                result2 = result;
    //                contours.at(0) = contours.at(index);
    //            }
    //            ++index;
    //            ++itc;
    //        }
    //    }
    //
    //    if (!contours.empty()) {
    //        cout << "Tamanho do contorno: " << contours.at(0).size() << endl;
    //        contours2.push_back(contours.at(0));
    //    }
    //
    //    //desenha contorno
    //    drawContours(bg, contours2, -1, CV_RGB(0, 250, 0), 2);
    //    drawContours(mask, contours2, -1, 255, CV_FILLED);


    //    std::vector<cv::Point> hull;
    //    convexHull(cv::Mat(contours[0]), hull);
    //
    //    // vector of keypoints
    //    std::vector<cv::KeyPoint> keypoints;
    //    // Construct the SURF feature detector object
    //    SurfFeatureDetector surf(2500.); // threshold
    //    // Detect the SURF features
    //    surf.detect(mask, keypoints);
    //    cv::drawKeypoints(mask, // original image
    //            keypoints, // vector of keypoints
    //            mask, // the resulting image
    //            cv::Scalar(255, 0, 0), // color of the points
    //            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS); //flag

    namedWindow("mascara_obtida");
    imshow("mascara_obtida", mask);
    imwrite("C://prints//mascara_obtida.jpg", mask);



    //    ----------------- PREPARAR PARA PEGAR RANGES
    //REDUCAO DE CORES - para melhorar o HUE (deve ser um parametro ajustavel)
    mao.copyTo(maoCorReduz);
    colorReduce(maoCorReduz, 32);
    namedWindow("reducaoCores", WINDOW_AUTOSIZE);
    imshow("reducaoCores", maoCorReduz);

    //normaliza os canais rgb individualmente
    equRGB(mao, maoEquRGB);
    namedWindow("maoEquRGB", WINDOW_AUTOSIZE);

    //     Imagem com RGB normalizado
    Mat mao_norm2(sz, CV_8UC3);
    normalizaRGB(mao, mao_norm2);
    namedWindow("normRGB", CV_WINDOW_AUTOSIZE);
    imshow("normRGB", mao_norm2);

    //---------------------------------------------------------------
    Scalar media_rgb, media_hsv, media_h, media_lab;
    Scalar desvioPadrao_rgb, desvioPadrao_hsv, desvioPadrao_h, desvioPadrao_lab;


    //CALCULO DA FAIXA DE RGB
    meanStdDev(mao, media_rgb, desvioPadrao_rgb, mask);
    cout << "------RGB------" << endl;
    cout << "Media B: " << media_rgb(0) << endl;
    cout << "Media G: " << media_rgb(1) << endl;
    cout << "Media R: " << media_rgb(2) << endl;
    cout << "Desvio B: " << desvioPadrao_rgb(0) << endl;
    cout << "Desvio G: " << desvioPadrao_rgb(1) << endl;
    cout << "Desvio R: " << desvioPadrao_rgb(2) << endl;
    Scalar low_rgb = media_rgb - (desvioPadrao_rgb);
    Scalar high_rgb = media_rgb + (desvioPadrao_rgb);

    //    inRange(mao, low_rgb, high_rgb, final_rgb);
    //    trataContorno(final_rgb, final_rgb);
    //    namedWindow("RGB", CV_WINDOW_AUTOSIZE);
    //    imshow("RGB", final_rgb);


    //CALCULO DA FAIXA DE HUE
    Mat h(sz, CV_8SC1);
    Mat mao_hsv;
    cvtColor(mao, mao_hsv, CV_BGR2HSV_FULL);
    vector<Mat> hsv_planes;
    split(mao_hsv, hsv_planes);
    hsv_planes[0].copyTo(h);
    erode(mask, mask, Mat(), Point(-1, -1), 3);
    meanStdDev(h, media_h, desvioPadrao_h, mask);
    Scalar high_h = (140);
    Scalar low_h = (80);


    //    CALCULO DA FAIXA DE LAB
    Mat lab_mao;
    cvtColor(mao, lab_mao, CV_BGR2Lab);
    meanStdDev(lab_mao, media_lab, desvioPadrao_lab, mask);
    cout << "------LAB------" << endl;
    cout << "Media L: " << media_lab(0) << endl;
    cout << "Media A: " << media_lab(1) << endl;
    cout << "Media B: " << media_lab(2) << endl;
    cout << "Desvio L: " << desvioPadrao_lab(0) << endl;
    cout << "Desvio A: " << desvioPadrao_lab(1) << endl;
    cout << "Desvio B: " << desvioPadrao_lab(2) << endl;
    Scalar low_lab = media_lab - (desvioPadrao_lab);
    Scalar high_lab = media_lab + (desvioPadrao_lab)+(desvioPadrao_lab);


    //---------------TEMPO DE EXECUCAO-------------------------------------------
    if (!capture.isOpened()) {
        cout << "Captura de video falhou!" << endl;
        return 1;
    }
    //    Mat frame;
    Mat frame_hsv;
    Mat temp2;
    Mat frame_h;
    Mat frameSub;
    Mat frame_lab;

    //LOOP DE CAPTURA
    while (c != 27) {
        Mat frame;
        Mat frameFull;
        Mat frame_mask(sz, CV_8UC1);
        if (!capture.read(frameFull)) {
            cout << "Falha na captura" << endl;
            break;
        }

        //EXIBE VIDEO
        Rect roi(0, 0, sz.width, sz.height);
        frame = frameFull(roi);
        rectangle(frameFull, roi, Scalar(0, 0, 0), 2);
        namedWindow("FRAME ROI", WINDOW_AUTOSIZE);
        imshow("FRAME ROI", frameFull);

        Mat bg(sz, CV_8UC3);
        rectangle(bg, roi, Scalar(255, 0, 0), 2);

        //--------------------------------------------------
        //SUBTRACAO FUNDO-FRAME
        subtracao(fundo, frame, frameSub);
        //        namedWindow("Frame Subtracao", WINDOW_AUTOSIZE);
        //        imshow("Frame Subtracao", resultSub);

        //LIMIARIZACAO FRAME
        threshold(frameSub, frameSub, 10, 255, CV_THRESH_BINARY);
        namedWindow("Frame Limiarizacao", WINDOW_AUTOSIZE);
        imshow("Frame Limiarizacao", resultSub);

        //OPERACOES FRAME
        operacoes(frameSub, 7, 2, 1);
        namedWindow("Frame Operacoes", WINDOW_AUTOSIZE);
        imshow("Frame Operacoes", resultSub);
        //RESULTADO SUBTRACAO FRAME
        //                trataContorno(resultSub, frame_mask,2);
        namedWindow("Frame Mascara Sub");
        imshow("Frame Mascara Sub", frameSub);
        //                resultSub.release();
        //                mask.release();

        //-------------------------------------------

        //Deteccao pela componente HUE
        cvtColor(frame, frame_hsv, CV_BGR2HSV_FULL);
        vector<Mat> hsv_canais;
        split(frame_hsv, hsv_canais);
        hsv_canais[0].copyTo(frame_h);

        inRange(frame_h, (0), (1), temp2);
        inRange(frame_h, low_h, high_h, frame_h);
        //        namedWindow("Frame HUE inver", CV_WINDOW_AUTOSIZE);
        //        imshow("Frame HUE inver", frame_h);
        bitwise_not(frame_h, frame_h);
        subtract(frame_h, temp2, frame_h);

        namedWindow("Frame HUE TESTE", CV_WINDOW_AUTOSIZE);
        imshow("Frame HUE TESTE", temp2);

        namedWindow("Frame HUE", CV_WINDOW_AUTOSIZE);
        imshow("Frame HUE", frame_h);
        operacoes(frame_h, 7, 2, 2);
        //        trataContorno(frame_h, frame_h);
        //        namedWindow("Frame HUE", CV_WINDOW_AUTOSIZE);
        //        imshow("Frame HUE", frame_h);




        //DETECCAO RGB ORIGINAL
        skinRGBDetect(frame, rgb_mask);
        namedWindow("Frame skinRGB", CV_WINDOW_AUTOSIZE);
        imshow("Frame skinRGB", rgb_mask);
        operacoes(rgb_mask, 7, 2, 2);
        //        trataContorno(rgb_mask, rgb_mask);
        //        namedWindow("Frame skinRGB", CV_WINDOW_AUTOSIZE);
        //        imshow("Frame skinRGB", rgb_mask);

        // DETECCAO POR MEDIA RGB
        inRange(frame, low_rgb, high_rgb, media_rgb_mask);
        namedWindow("Frame media RGB", CV_WINDOW_AUTOSIZE);
        imshow("Frame media RGB", media_rgb_mask);
        operacoes(media_rgb_mask, 7, 2, 2);
        //        trataContorno(final_rgb, final_rgb);
        //        namedWindow("Frame media RGB", CV_WINDOW_AUTOSIZE);
        //        imshow("Frame media RGB", media_rgb_mask);

        //rgb media + skin detect
        add(rgb_mask, media_rgb_mask, final_rgb);
        namedWindow("Frame Final RGB", CV_WINDOW_AUTOSIZE);
        imshow("Frame Final RGB", final_rgb);
        operacoes(final_rgb, 7, 2, 2);
        //        namedWindow("Frame Final RGB", CV_WINDOW_AUTOSIZE);
        //        imshow("Frame Final RGB", final_rgb);



        // DETECCAO POR MEDIA LAB
        cvtColor(frame, frame_lab, CV_BGR2Lab);
        inRange(frame_lab, low_lab, high_lab, frame_lab);
        namedWindow("Frame LAB", CV_WINDOW_AUTOSIZE);
        imshow("Frame LAB", frame_lab);
        operacoes(frame_lab, 7, 2, 2);
        //                trataContorno(frame_lab, frame_lab);
        //        namedWindow("Frame LAB", CV_WINDOW_AUTOSIZE);
        //        imshow("Frame LAB", frame_lab);

        //-----------------------------------------------------------
        //RECONHECIMENTO
        Mat segmentado;
        frameSub.copyTo(segmentado);

        findContours(segmentado, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        double result = 0, result2 = 0, area;
        //considera apenas contornos com o perimetro dentro da faixa especificada
        int cmin = 20000; //400; // minimum contour length
        int cmax = 100000; //2800; // maximum contour length    
        vector<vector<Point> >::iterator itc = contours.begin();
        int index = 0;
        while (itc != contours.end()) {
            area = fabs(contourArea(contours.at(index)));
            if (area < cmin || area > cmax)
                itc = contours.erase(itc);
            else {
                if (area > result) {
                    result = area;
                    contours.at(0) = contours.at(index);
                }
                ++itc;
                ++index;
            }
        }
        //                        comparaContorno("amostra5.jpg", contours);
        vector<int> hull;
        vector<Vec4i> defects;
        int i = 0;
        if (!contours.empty()) {
            //            cout << "PERIMETRO: " << contours[0].size() << endl;
            //            cout << "AREA: " << contourArea(contours[0], false) << endl;
            vector<Point> pontos = contours[0];
            convexHull(pontos, hull);

            //desenha hull
            int hullcount = (int) hull.size();
            Point pt0 = pontos[hull[hullcount - 1]];
            for (i = 0; i < hullcount; i++) {
                Point pt = pontos[hull[i]];
                line(frame, pt0, pt, Scalar(0, 255, 0), 1, CV_AA);
                pt0 = pt;
            }

            //Identifica e conta os Defects
            int defectCount = 0;
            int profundidadeDefect;
            int ptMaisDistante;
            if (pontos.size() > 3 & hull.size() > 2) {
                convexityDefects(pontos, hull, defects);
                for (i = 0; i < defects.size(); i++) {
                    profundidadeDefect = defects[i][3];
                    ptMaisDistante = defects[i][2];
                    if (profundidadeDefect / 256 > 40) { //profundidade do defect
                        //cout << "Profundidade: " << profundidadeDefect/256 << endl;
                        circle(frameFull, pontos[ptMaisDistante], 7, Scalar(0, 0, 255), 2);
                        defectCount++;
                    }
                    //cout << "Numero de defects: " defectCount << endl;
                }
            }

            //            namedWindow("Frame HULL", CV_WINDOW_AUTOSIZE);
            //            imshow("Frame HULL", frame);
            reconheceGesto(defectCount, contours);
        }

        drawContours(frameFull, contours, 0, (0, 0, 255), 2);
        namedWindow("Frame CONTORNO", CV_WINDOW_AUTOSIZE);
        imshow("Frame CONTORNO", frameFull);

        //TODO: Desenhar convex hull
        //TODO enquanto nao achar contorno c/ tamanho razoavel, diminuir o limiar
        c = waitKey(10);

        if (c == 102) {
            imwrite("C://prints//frameFull.jpg", frameFull);
            imwrite("C://prints//frame_lab.jpg", frame_lab);
            imwrite("C://prints//media_rgb_mask.jpg", media_rgb_mask);
            imwrite("C://prints//final_rgb.jpg", final_rgb);
            imwrite("C://prints//rgb_mask.jpg", rgb_mask);
            imwrite("C://prints//frame_h.jpg", frame_h);
            imwrite("C://prints//temp2.jpg", temp2);
            imwrite("C://prints//frameSub.jpg", frameSub);
        }

        //        if(c==32){
        //        
        //        }

        frame_mask.release();
        media_rgb_mask.release();
        frameFull.release();
    }
}
