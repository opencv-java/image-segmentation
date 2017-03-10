package it.polito.elite.teaching.cv;
	
import org.opencv.core.Core;

import javafx.application.Application;
import javafx.event.EventHandler;
import javafx.stage.Stage;
import javafx.stage.WindowEvent;
import javafx.scene.Scene;
import javafx.scene.layout.BorderPane;
import javafx.fxml.FXMLLoader;

public class ImageSegmentation extends Application
{
	/**
	 * The main class for a JavaFX application. It creates and handle the main
	 * window with its resources (style, graphics, etc.).
	 * 
	 * This application apply the Canny filter to the camera video stream or try
	 * to remove a uniform background with the erosion and dilation operators.
	 * 
	 * @author <a href="mailto:luigi.derussis@polito.it">Luigi De Russis</a>
	 * @author <a href="mailto:alberto.cannavo@polito.it">Alberto Cannavò</a>
	 * @version 2.0 (2017-03-10)
	 * @since 1.0 (2013-12-20)
	 * 
	 */
	@Override
	public void start(Stage primaryStage)
	{
		try
		{
			// load the FXML resource
			FXMLLoader loader = new FXMLLoader(getClass().getResource("ImageSeg.fxml"));
			BorderPane root = (BorderPane) loader.load();
									
			// set a whitesmoke background
			root.setStyle("-fx-background-color: whitesmoke;");
			// create and style a scene
			Scene scene = new Scene(root, 800, 600);
			scene.getStylesheets().add(getClass().getResource("application.css").toExternalForm());
			// create the stage with the given title and the previously created
			// scene
			primaryStage.setTitle("Image Segmentation");
			primaryStage.setScene(scene);
			
			// show the GUI
			primaryStage.show();
			
			// get the controller
			ImageSegController controller = loader.getController();			
			controller.init();
			
			// set the proper behavior on closing the application
			primaryStage.setOnCloseRequest((new EventHandler<WindowEvent>() {
				public void handle(WindowEvent we)
				{
					controller.setClosed();
				}
			}));
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args)
	{
		// load the native OpenCV library
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		launch(args);
	}
}
