package it.polito.elite.teaching.cv;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

import it.polito.elite.teaching.cv.utils.Utils;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

/**
 * The controller associated with the only view of our application. The
 * application logic is implemented here. It handles the button for
 * starting/stopping the camera, the acquired video stream, the relative
 * controls and the image segmentation process.
 * 
 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
 * @author <a href="mailto:alberto.cannavo@polito.it">Alberto Cannavò</a>
 * @version 2.0 (2017-03-10)
 * @since 1.0 (2013-12-20)
 * 
 */

public class ImageSegController
{
	
	// FXML buttons
	@FXML
	private Button cameraButton;
	// the FXML area for showing the current frame
	@FXML
	private ImageView originalFrame;
	// checkbox for enabling/disabling Canny
	@FXML
	private CheckBox canny;
	// canny threshold value
	@FXML
	private Slider threshold;
	// checkbox for enabling/disabling background removal
	@FXML
	private CheckBox dilateErode;
	// inverse the threshold value for background removal
	@FXML
	private CheckBox inverse;
	
	// a timer for acquiring the video stream
	private ScheduledExecutorService timer;
	// the OpenCV object that performs the video capture
	private VideoCapture capture = new VideoCapture();
	// a flag to change the button behavior
	private boolean cameraActive;
	
	Point clickedPoint = new Point(0, 0);
	Mat oldFrame;
	
	protected void init()
	{
		this.threshold.setShowTickLabels(true);
	}
	
	/**
	 * The action triggered by pushing the button on the GUI
	 */
	@FXML
	protected void startCamera()
	{
		// set a fixed width for the frame
		originalFrame.setFitWidth(380);
		// preserve image ratio
		originalFrame.setPreserveRatio(true);
		
		// mouse listener
		originalFrame.setOnMouseClicked(e -> {
			System.out.println("[" + e.getX() + ", " + e.getY() + "]");
			clickedPoint.x = e.getX();
			clickedPoint.y = e.getY();
		});
		
		if (!this.cameraActive)
		{
			// disable setting checkboxes
			this.canny.setDisable(true);
			this.dilateErode.setDisable(true);
			
			// start the video capture
			this.capture.open(0);
			
			// is the video stream available?
			if (this.capture.isOpened())
			{
				this.cameraActive = true;
				
				// grab a frame every 33 ms (30 frames/sec)
				Runnable frameGrabber = new Runnable() {
					
					@Override
					public void run()
					{
						// effectively grab and process a single frame
						Mat frame = grabFrame();
						// convert and show the frame
						Image imageToShow = Utils.mat2Image(frame);
						updateImageView(originalFrame, imageToShow);
					}
				};
				
				this.timer = Executors.newSingleThreadScheduledExecutor();
				this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
				
				// update the button content
				this.cameraButton.setText("Stop Camera");
			}
			else
			{
				// log the error
				System.err.println("Failed to open the camera connection...");
			}
		}
		else
		{
			// the camera is not active at this point
			this.cameraActive = false;
			// update again the button content
			this.cameraButton.setText("Start Camera");
			// enable setting checkboxes
			this.canny.setDisable(false);
			this.dilateErode.setDisable(false);
			
			// stop the timer
			this.stopAcquisition();
		}
	}
	
	/**
	 * Get a frame from the opened video stream (if any)
	 * 
	 * @return the {@link Image} to show
	 */
	private Mat grabFrame()
	{
		Mat frame = new Mat();
		
		// check if the capture is open
		if (this.capture.isOpened())
		{
			try
			{
				// read the current frame
				this.capture.read(frame);
				
				// if the frame is not empty, process it
				if (!frame.empty())
				{
					// handle edge detection
					if (this.canny.isSelected())
					{
						frame = this.doCanny(frame);
						//frame = this.doSobel(frame);
					}
					// foreground detection
					else if (this.dilateErode.isSelected())
					{
						// Es. 2.1
						// frame = this.doBackgroundRemovalFloodFill(frame);
						// Es. 2.2
						frame = this.doBackgroundRemovalAbsDiff(frame);
						// Es. 2.3
						// frame = this.doBackgroundRemoval(frame);
						
					}
					
				}
				
			}
			catch (Exception e)
			{
				// log the (full) error
				System.err.print("Exception in the image elaboration...");
				e.printStackTrace();
			}
		}
		
		return frame;
	}
	
	private Mat doBackgroundRemovalAbsDiff(Mat currFrame)
	{
		Mat greyImage = new Mat();
		Mat foregroundImage = new Mat();
		
		if (oldFrame == null)
			oldFrame = currFrame;
		
		Core.absdiff(currFrame, oldFrame, foregroundImage);
		Imgproc.cvtColor(foregroundImage, greyImage, Imgproc.COLOR_BGR2GRAY);
		
		int thresh_type = Imgproc.THRESH_BINARY_INV;
		if (this.inverse.isSelected())
			thresh_type = Imgproc.THRESH_BINARY;
		
		Imgproc.threshold(greyImage, greyImage, 10, 255, thresh_type);
		currFrame.copyTo(foregroundImage, greyImage);
		
		oldFrame = currFrame;
		return foregroundImage;
		
	}
	
	private Mat doBackgroundRemovalFloodFill(Mat frame)
	{
		
		Scalar newVal = new Scalar(255, 255, 255);
		Scalar loDiff = new Scalar(50, 50, 50);
		Scalar upDiff = new Scalar(50, 50, 50);
		Point seedPoint = clickedPoint;
		Mat mask = new Mat();
		Rect rect = new Rect();
		
		// Imgproc.floodFill(frame, mask, seedPoint, newVal);
		Imgproc.floodFill(frame, mask, seedPoint, newVal, rect, loDiff, upDiff, Imgproc.FLOODFILL_FIXED_RANGE);
		
		return frame;
	}
	
	/**
	 * Perform the operations needed for removing a uniform background
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image with only foreground objects
	 */
	private Mat doBackgroundRemoval(Mat frame)
	{
		// init
		Mat hsvImg = new Mat();
		List<Mat> hsvPlanes = new ArrayList<>();
		Mat thresholdImg = new Mat();
		
		int thresh_type = Imgproc.THRESH_BINARY_INV;
		if (this.inverse.isSelected())
			thresh_type = Imgproc.THRESH_BINARY;
		
		// threshold the image with the average hue value
		hsvImg.create(frame.size(), CvType.CV_8U);
		Imgproc.cvtColor(frame, hsvImg, Imgproc.COLOR_BGR2HSV);
		Core.split(hsvImg, hsvPlanes);
		
		// get the average hue value of the image
		double threshValue = this.getHistAverage(hsvImg, hsvPlanes.get(0));
		
		Imgproc.threshold(hsvPlanes.get(0), thresholdImg, threshValue, 179.0, thresh_type);
		
		Imgproc.blur(thresholdImg, thresholdImg, new Size(5, 5));
		
		// dilate to fill gaps, erode to smooth edges
		Imgproc.dilate(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 1);
		Imgproc.erode(thresholdImg, thresholdImg, new Mat(), new Point(-1, -1), 3);
		
		Imgproc.threshold(thresholdImg, thresholdImg, threshValue, 179.0, Imgproc.THRESH_BINARY);
		
		// create the new image
		Mat foreground = new Mat(frame.size(), CvType.CV_8UC3, new Scalar(255, 255, 255));
		frame.copyTo(foreground, thresholdImg);
		
		return foreground;
	}
	
	/**
	 * Get the average hue value of the image starting from its Hue channel
	 * histogram
	 * 
	 * @param hsvImg
	 *            the current frame in HSV
	 * @param hueValues
	 *            the Hue component of the current frame
	 * @return the average Hue value
	 */
	private double getHistAverage(Mat hsvImg, Mat hueValues)
	{
		// init
		double average = 0.0;
		Mat hist_hue = new Mat();
		// 0-180: range of Hue values
		MatOfInt histSize = new MatOfInt(180);
		List<Mat> hue = new ArrayList<>();
		hue.add(hueValues);
		
		// compute the histogram
		Imgproc.calcHist(hue, new MatOfInt(0), new Mat(), hist_hue, histSize, new MatOfFloat(0, 179));
		
		// get the average Hue value of the image
		// (sum(bin(h)*h))/(image-height*image-width)
		// -----------------
		// equivalent to get the hue of each pixel in the image, add them, and
		// divide for the image size (height and width)
		for (int h = 0; h < 180; h++)
		{
			// for each bin, get its value and multiply it for the corresponding
			// hue
			average += (hist_hue.get(h, 0)[0] * h);
		}
		
		// return the average hue of the image
		return average = average / hsvImg.size().height / hsvImg.size().width;
	}
	
	/**
	 * Apply Canny
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image elaborated with Canny
	 */
	private Mat doCanny(Mat frame)
	{
		// init
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat();
		
		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		
		// reduce noise with a 3x3 kernel
		Imgproc.blur(grayImage, detectedEdges, new Size(3, 3));
		
		// canny detector, with ratio of lower:upper threshold of 3:1
		Imgproc.Canny(detectedEdges, detectedEdges, this.threshold.getValue(), this.threshold.getValue() * 3);
		
		// using Canny's output as a mask, display the result
		Mat dest = new Mat();
		frame.copyTo(dest, detectedEdges);
		
		return dest;
	}
	
	/**
	 * Apply Sobel
	 * 
	 * @param frame
	 *            the current frame
	 * @return an image elaborated with Sobel derivation
	 */
	private Mat doSobel(Mat frame)
	{
		// init
		Mat grayImage = new Mat();
		Mat detectedEdges = new Mat();
		int scale = 1;
		int delta = 0;
		int ddepth = CvType.CV_16S;
		Mat grad_x = new Mat();
		Mat grad_y = new Mat();
		Mat abs_grad_x = new Mat();
		Mat abs_grad_y = new Mat();
		
		// reduce noise with a 3x3 kernel
		Imgproc.GaussianBlur(frame, frame, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT);
		
		// convert to grayscale
		Imgproc.cvtColor(frame, grayImage, Imgproc.COLOR_BGR2GRAY);
		
		// Gradient X
		// Imgproc.Sobel(grayImage, grad_x, ddepth, 1, 0, 3, scale,
		// this.threshold.getValue(), Core.BORDER_DEFAULT );
		Imgproc.Sobel(grayImage, grad_x, ddepth, 1, 0);
		Core.convertScaleAbs(grad_x, abs_grad_x);
		
		// Gradient Y
		// Imgproc.Sobel(grayImage, grad_y, ddepth, 0, 1, 3, scale,
		// this.threshold.getValue(), Core.BORDER_DEFAULT );
		Imgproc.Sobel(grayImage, grad_y, ddepth, 0, 1);
		Core.convertScaleAbs(grad_y, abs_grad_y);
		
		// Total Gradient (approximate)
		Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, detectedEdges);
		// Core.addWeighted(grad_x, 0.5, grad_y, 0.5, 0, detectedEdges);
		
		return detectedEdges;
		
	}
	
	/**
	 * Action triggered when the Canny checkbox is selected
	 * 
	 */
	@FXML
	protected void cannySelected()
	{
		// check whether the other checkbox is selected and deselect it
		if (this.dilateErode.isSelected())
		{
			this.dilateErode.setSelected(false);
			this.inverse.setDisable(true);
		}
		
		// enable the threshold slider
		if (this.canny.isSelected())
			this.threshold.setDisable(false);
		else
			this.threshold.setDisable(true);
		
		// now the capture can start
		this.cameraButton.setDisable(false);
	}
	
	/**
	 * Action triggered when the "background removal" checkbox is selected
	 */
	@FXML
	protected void dilateErodeSelected()
	{
		// check whether the canny checkbox is selected, deselect it and disable
		// its slider
		if (this.canny.isSelected())
		{
			this.canny.setSelected(false);
			this.threshold.setDisable(true);
		}
		
		if (this.dilateErode.isSelected())
			this.inverse.setDisable(false);
		else
			this.inverse.setDisable(true);
		
		// now the capture can start
		this.cameraButton.setDisable(false);
	}
	
	/**
	 * Stop the acquisition from the camera and release all the resources
	 */
	private void stopAcquisition()
	{
		if (this.timer != null && !this.timer.isShutdown())
		{
			try
			{
				// stop the timer
				this.timer.shutdown();
				this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
			}
			catch (InterruptedException e)
			{
				// log any exception
				System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
			}
		}
		
		if (this.capture.isOpened())
		{
			// release the camera
			this.capture.release();
		}
	}
	
	/**
	 * Update the {@link ImageView} in the JavaFX main thread
	 * 
	 * @param view
	 *            the {@link ImageView} to update
	 * @param image
	 *            the {@link Image} to show
	 */
	private void updateImageView(ImageView view, Image image)
	{
		Utils.onFXThread(view.imageProperty(), image);
	}
	
	/**
	 * On application close, stop the acquisition from the camera
	 */
	protected void setClosed()
	{
		this.stopAcquisition();
	}
	
}
