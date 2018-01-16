#include "drivers/xbgsub.h"

#include "include/xparameters.h"
#include <chrono>
#include <string.h>
#include <fstream>
#include <iostream>

// #include "detection/MyTypes.h"
#include "detection/ClientUDP.h"
// #include "detection/MyTypes.h"
#include "detection/BGSDetector.h"
#include <csignal>

#define RGB_TX_BASE_ADDR  0x10000000
#define MASK_BASE_ADDR    0x10100000
#define TX_BASE_ADDR      0x11000000
#define RX_BASE_ADDR      0x11800000
#define BG_MODEL          0x13000000
#define DDR_RANGE         0x00800000

#define AXILITES_BASEADDR 0x43C00000
#define CRTL_BUS_BASEADDR 0x43C10000
#define AXILITE_RANGE     0x0000FFFF



using namespace cv;
using namespace std;


/***************** Global Variables *********************/


XBgsub backsub;

int fdIP;
int fd; // A file descriptor to the video device
int type;
// uint8_t * ybuffer = new uint8_t[N];

uint8_t * src; 
uint8_t * dst; 




int backsub_init(XBgsub * backsub_ptr){
    backsub_ptr->Axilites_BaseAddress = (u32)mmap(NULL, AXILITE_RANGE, PROT_READ|PROT_WRITE, MAP_SHARED, fdIP, XPAR_BGSUB_0_S_AXI_AXILITES_BASEADDR);
    backsub_ptr->Crtl_bus_BaseAddress = (u32)mmap(NULL, AXILITE_RANGE, PROT_READ|PROT_WRITE, MAP_SHARED, fdIP, XPAR_XBGSUB_0_S_AXI_CRTL_BUS_BASEADDR);
    backsub_ptr->IsReady = XIL_COMPONENT_IS_READY;
    return 0;
}

void backsub_rel(XBgsub * backsub_ptr){
    munmap((void*)backsub_ptr->Axilites_BaseAddress, AXILITE_RANGE);
    munmap((void*)backsub_ptr->Crtl_bus_BaseAddress, AXILITE_RANGE);
}

void backsub_config(bool ini) {
    XBgsub_Set_frame_in(&backsub,(u32)TX_BASE_ADDR);
    XBgsub_Set_frame_out(&backsub,(u32)RX_BASE_ADDR);
    XBgsub_Set_init(&backsub, ini);
    XBgsub_Set_bgmodel(&backsub, (u32)BG_MODEL);
}

void print_config() {
    printf("Is Ready = %d \n", XBgsub_IsReady(&backsub));
    printf("Frame in = %X \n", XBgsub_Get_frame_in(&backsub));
    printf("Frame out = %X \n", XBgsub_Get_frame_out(&backsub));
    printf("Init = %d \n", XBgsub_Get_init(&backsub));
}


void signalHandler( int signum ) {
    cout << "Interrupt signal (" << signum << ") received.\n";

    // cleanup and close up stuff here
    // terminate program

    //Release IP Core
    backsub_rel(&backsub);
   
    munmap((void*)src, DDR_RANGE);
    munmap((void*)dst, DDR_RANGE);


    close(fdIP);

    exit(signum);
}


int main(int argc, char *argv[]) {
    signal(SIGINT, signalHandler);

    // Initialization communication link
    boost::asio::io_service io_service;
    ClientUDP client(io_service,"10.10.21.49",8080);
    uint16_t frameNo=0;
    const uint8_t cameraID = 0;

    // Initializing IP Core Starts here .........................
    fdIP = open ("/dev/mem", O_RDWR);
    if (fdIP < 1) {
        perror(argv[0]);
        return -1;
    }

    VideoCapture cap(argv[1]);
    cap.set(CV_CAP_PROP_FRAME_WIDTH,320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT,240);
    cap.set(CV_CAP_PROP_FPS,30);
    cap.set(CV_CAP_PROP_CONVERT_RGB,true);
   // cap.set(CV_CAP_PROP_AUTOFOCUS, 0);

    src = (uint8_t*)mmap(NULL, DDR_RANGE,PROT_READ|PROT_WRITE, MAP_SHARED, fdIP, TX_BASE_ADDR); 
    dst = (uint8_t*)mmap(NULL, DDR_RANGE,PROT_EXEC|PROT_READ|PROT_WRITE, MAP_SHARED, fdIP, RX_BASE_ADDR);


    printf("init begin\n");

    if(backsub_init(&backsub)==0) {
        printf("Backsub IP Core Initialized!\n");
    }

    // Initializing IP Core Ends here .........................

    
    BGSDetector detector(30,
                          BGS_HW,
                          false,
                          "./pca_coeff.xml",
                          false);

    /***************************** Begin looping here *********************/
//    auto begin = std::chrono::high_resolution_clock::now();
    bool isFirst = true;
    Mat img, grey;

    for (;;){
        // Queue the buffer
        //auto begin = std::chrono::high_resolution_clock::now();

        backsub_config(isFirst);
        if(isFirst) isFirst = false;

    auto begin = std::chrono::high_resolution_clock::now();
    cap>>img;
    auto begin2 = std::chrono::high_resolution_clock::now();
    if(!img.data) break;
        cv::cvtColor(img, grey, CV_BGR2GRAY);
        memcpy(src,grey.data,76800);
        //auto begin2 = std::chrono::high_resolution_clock::now();

        XBgsub_Start(&backsub);
        while(!XBgsub_IsDone(&backsub));

        auto end2 = std::chrono::high_resolution_clock::now();
        Mat mask = Mat(240, 320, CV_8UC1); 
        memcpy(mask.data,dst,76800);

        std::vector<cv::Rect> detections = detector.detect(mask);
            int len = detections.size();
            if (len>10){
                len = 10;
            }
            printf("detections: %d\n",len);
            int det =0;
             

        auto end3 = std::chrono::high_resolution_clock::now();
        Frame frame;
        frame.frameNo = frameNo;
        frame.cameraID = cameraID;
        frame.detections.clear();
        frame.histograms.clear();
        for(int q=0;q<len;q++)
        {
            BoundingBox bbox;
            bbox.x = detections[q].x;
            bbox.y = detections[q].y;
            bbox.width = detections[q].width;
            bbox.height = detections[q].height;
            frame.detections.push_back(bbox);
      
            frame.histograms.push_back(detector.histograms[q]);
        }
        frameNo++;
        frame.setMask(detector.mask);
        frame.set_now();
        auto end4 = std::chrono::high_resolution_clock::now();
        client.send(frame);

        auto end = std::chrono::high_resolution_clock::now();

    printf("Elapsed time capture : %lld us\n",std::chrono::duration_cast<std::chrono::microseconds>(begin2-begin).count());
    printf("Elapsed time backsub : %lld us\n",std::chrono::duration_cast<std::chrono::microseconds>(end2-begin2).count());
    printf("Elapsed time opencv  : %lld us\n",std::chrono::duration_cast<std::chrono::microseconds>(end3-end2).count());
    printf("Elapsed time send    : %lld us\n",std::chrono::duration_cast<std::chrono::microseconds>(end-end4).count());
    printf("Elapsed time total   : %lld us\n",std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count());
    
}


    //Release IP Core
    backsub_rel(&backsub);

    munmap((void*)src, DDR_RANGE);
    munmap((void*)dst, DDR_RANGE);

    close(fdIP);
     
    printf("Device unmapped\n");

    return 0;
}

