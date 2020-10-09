#include "vita.hpp"


int main() try
{
    pthread_t thread_extract, thread_process;
    cv::namedWindow( "Facial Landmark Detection", cv::WINDOW_NORMAL );
    cv::setWindowProperty("Facial Landmark Detection", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

    vita vita;
    
    vita.capture_frame();
    
    pthread_create(&thread_process, NULL, vita.start_process_thread, (void*)&vita);
    pthread_create(&thread_extract, NULL, vita.start_extract_thread, (void*)&vita);
    
    imshow("Facial Landmark Detection", vita.image);
    
    while(1)
    {
        if (waitKey(1) == 32) //Space to start
        {
            if(vita.start_flg == 1)
            {
                vita.start_flg = 0;  // Running
            }
            else
            {
                vita.start_flg = 1;  //Pause
            }
            cout<<"space pressed"<<endl;
        }
        
        vita.capture_frame();
        imshow("Facial Landmark Detection", vita.image);
        
        if (waitKey(1) == 27)  //ESC to exit
        {
            vita.start_flg = 1;
            vita.flg = true;
            pthread_join(thread_process, NULL);
            pthread_join(thread_extract, NULL);
            break;
        }
    }
    
    return 0;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
