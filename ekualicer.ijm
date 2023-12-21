// This macro runs the equalization procedure per slice using ROIs for every slice which are obtained from a rough threshold of the material
// and background that you want to adjust to a certain gray level. The volumes for material and background need to be created beforehand and
//saved in the same folder as the volume to equalize.
// It also allows creating a fixed BACKGROUND or a fix MATERIAL or both ROIs which need to be named with a number code at the beginningof the name 
//for proper sorting in the macro. Ej. 00_MAT and 01_BKG.


//debug;
//Equalization start:
/* 1.Jump to slice start
 *  2.Duplicate it
 *  3.equalize it
 *  4.build a new stack with this slice as nr one
 *  5. do this until slice end
 *  6.apply the changes applied for slices start to slices before start and that concatenate it
 *  7.aplly changes of slices end to the rest of the volume and append ...
 */
 //see https://imagej.nih.gov/ij/developer/macro/functions.html for built in functions.


run("ROI Manager...");  //this is the first windows it opens

setBatchMode(true); //Controls whether images are visible or hidden during macro execution.

//Reference input:
ref_mat_original =150;
ref_bkg_original = 60;
//Correction of the reference:
ref_mat = ref_mat_original; // Equivalent to B' in excel file
ref_bkg = ref_bkg_original ;  // Equivalent to A' in excel file

//Equalization Methods: First we define the functions we are going to use
// I understand that the following variables are equivalent
// mat = B (the gray level of material in the 16b file)
// bkg = A (the grey level of bkg in 16b file)
// E3 = C in the excel file
function equalX(mat, bkg, ref_mat, ref_bkg){
	//E3 = (ref_mat + 1)/(ref_bkg + 1); old variable. New one is
	E3 = (mat-bkg)/(ref_mat-ref_bkg);
	//X =  (E3 * bkg + E3 - mat -1)/(E3-1);
	X = bkg - ref_bkg*E3;
	X = round(X);
	return(X);
}
function equalY(mat, bkg, ref_mat, ref_bkg, X){
	//E3 = (ref_mat + 1)/(ref_bkg + 1);
	//Y =  256 * (mat - X + 1)/(ref_mat + 1) + X - 1;
	E3 = (mat-bkg)/(ref_mat-ref_bkg);
	Y = bkg + E3*(255-ref_bkg);
	Y = round(Y);
	return(Y);
}

//Initialization:
print("\\Clear");
run("Clear Results");
//Initial threshold and tolerance value:
delta = 0.6 ;
t_mat = 0.88; //set at 40%. Values coincide with excel file
t_bkg = 0.88;

//User inputs:

Dialog.create("Equalization Settings");
Dialog.addNumber("Reference material:", ref_mat_original);
Dialog.addNumber("Reference background:", ref_bkg_original);
Dialog.addNumber("Threshold material:", t_mat);
Dialog.addNumber("Threshold background:", t_bkg);
Dialog.addNumber("Tolerance:", delta);
Dialog.addMessage("Slices to be analyzed: ");
Dialog.addNumber("At the beginning:", 1);
Dialog.addNumber("At the end:", 1400);
Dialog.addNumber("Fixed Background ROI", 1);
Dialog.addNumber("Fixed Material ROI", 1);
Dialog.show();

ref_mat_original = Dialog.getNumber();
ref_bkg_original = Dialog.getNumber();
t_mat = Dialog.getNumber();
t_bkg = Dialog.getNumber();
delta = Dialog.getNumber();
start_slice = Dialog.getNumber();
end_slice = Dialog.getNumber();
fix_bkg_ROI = Dialog.getNumber();
fix_mat_ROI = Dialog.getNumber();

print("The stack will be analized from slice " + start_slice + " to slice " + end_slice + ".");
print("Tolerance = " + delta);
print("Threshold material = " + t_mat);
print("Threshold background = " + t_bkg);

//Corection of the reference:
ref_mat = ref_mat_original; // Equivalent to B' in excel file
ref_bkg = ref_bkg_original;  // Equivalent to A' in excel file

// Step 1.Load the 16bit volume to be equalized:

print("Please specify the 16 bit volume...");
run("Raw...");
print("Volume loaded properly.");

// Step 1.2 Stack data: Obtain some information of the loaded volume.
Target_ID = getImageID();
Target_Title = getTitle();
Target_Title = replace(Target_Title, ".raw","");
Target_Directory = getDirectory("image"); //getDir("image") - Returns the path to the directory that the active image was loaded from
Target_Slices = nSlices; // nSlices: Returns the number of images in the current stack. Returns 1 if the current image is not a stack. 
						 // The parentheses "()" are optional. See also: getSliceNumber, getDimensions. 
Width = getWidth();
Height = getHeight();
//........

error = false;

if (end_slice > Target_Slices){
	Dialog.create("Equalization done!");
	Dialog.addMessage("Macro Error:");
	Dialog.addMessage("Slice end number higher than stack length.");
	Dialog.show();
	error = true;
	} else if (start_slice > Target_Slices) {
		Dialog.create("Equalization done!");
		Dialog.addMessage("Macro Error:");
		Dialog.addMessage("Slice start number higher than stack length.");
		Dialog.show();
		error = true;
		}
print("Volume being analyzed : " + Target_Title);

// Step 1. Load the 8bit volumes for ROI creation. Depending on the number of fixed ROIs, we are going to create the ROIs for each slice from the bkg_vol and the mat_vol, or just load a fixed ROIs for its use.

if (!fix_mat_ROI){
	print("Please specify the 8 bit volume of material...");
	run("Raw...");
	print("Volume loaded properly.");
	Mat_Stack_ID = getImageID(); //Get the ID of the Mat_stack
	Mat_Stack = getTitle();
} else {
	roiManager("Open", Target_Directory + "00_MAT.roi"); // ROI 0
}

if (!fix_bkg_ROI){
	print("Please specify the 8 bit volume of bkg...");
	run("Raw...");
	print("Volume loaded properly.");
	Bkg_Stack_ID = getImageID(); //Get the ID of the Bkg_stack
	Bkg_Stack = getTitle();
} else {
	roiManager("Open", Target_Directory + "01_BKG.roi"); //ROI 1
}

// Step 1.3 Creating the new stack:
newImage("Eq_Stack", "8-bit black", Width, Height, 1); //newImage(title, type, width, height, depth): Opens a new image or 
													   //stack using the name title
New_Stack_ID = getImageID(); //Get the ID of the Eq_stack
New_Stack = getTitle();


// Step 2.0 Iterating trough the stack:

for (slice=start_slice; slice<=end_slice; slice++) {
	if (error){
		print("Macro aborted due to errors...");
		print("Please check the inputs and try again.");
		break;
		}
	print("\\Clear");
	print("Progress : " + ((slice-start_slice)/Target_Slices*100)+ "%" );

	//0. Create the ROIs from the mat and bkg volumes

	//Create the ROI for material
	if (!fix_mat_ROI){
		selectImage(Mat_Stack_ID);
		setSlice(slice);
		run("Create Selection");
		roiManager("Add");
		if ( fix_bkg_ROI) {
			roiManager("Select", 1);
		} else 	{
			roiManager("Select", 0);
		}			
		roiManager("Rename", "00_MAT");
		run("Select None");
	}
	
	//Create the ROI for bakground
	if (!fix_bkg_ROI){
		selectImage(Bkg_Stack_ID);
		setSlice(slice);
		run("Create Selection");
		roiManager("Add");
		roiManager("Select", 1);
		roiManager("Rename", "01_BKG");
		run("Select None");
	}
	
	//Organize the ROIs
	if (fix_mat_ROI || fix_bkg_ROI) {
		roiManager("Sort");
	}
	
	//1.Obtain the slice
	selectImage(Target_ID);
	run("Duplicate...", "Slice_" + slice + " duplicate range=" + slice + "-" + slice);
	Slice_ID = getImageID();
	selectImage(Slice_ID);
	//2.Equalize it
	
	//2.1 Material 16bit values:
	roiManager("Select", 0);
	nBins = 256;
	getMinAndMax(hMin, hMax);
	getHistogram(values,counts,nBins);  //,hMin,hMax);
	
	//2.1.1 Max material values:
	max_mat = 0;
	for (i = 0; i < counts.length;i++){
		if(counts[i] > max_mat){
			max_mat = counts[i]; 	// maximum value of the material peak
		}
	}
	
	//2.1.2 Threshold and average value of the material:
	treshold_mat = max_mat*t_mat;
	counter = 0;
	sum_mat = 0;
	for(i = 0; i < counts.length; i++){
		if(counts[i] > treshold_mat){
			counter = counter +counts[i];
			sum_mat = sum_mat + counts[i]*values[i];	
			}
		}
	mat_val = sum_mat / counter ;		// position of the maximum value of the material peak. Important value
	//print("16bit material average:\n", mat_val);
	
	//2.2 Background
	roiManager("Select", 1);
	getMinAndMax(hMin, hMax);
	getHistogram(values,counts,nBins);  // ,hMin,hMax);
	
	//2.2.1 Max background value:
	max_bkg = 0;
	for (i = 0; i < counts.length;i++){
		if(counts[i] > max_bkg){
			max_bkg = counts[i];
		}
	}
	
	//2.2.2. Threshold and average values of the background: 
	treshold_bkg = max_bkg*t_bkg;
	counter = 0;
	sum_bkg = 0;
	for(i = 0; i < counts.length; i++){
		if(counts[i] > treshold_bkg){
			counter = counter +counts[i];
			sum_bkg = sum_bkg + counts[i]*values[i];	
			}
	}
	bkg_val = sum_bkg / counter ;
	
	//print("16bit background average:\n" + bkg_val);
	selectImage(Slice_ID);
	close();
	
	//3.0 Equalization loop:
	
	//3.1 Intialization of the values:
	delta_mat = 2;  //difference between ref_mat_original and ref_mat calculated
	delta_bkg = 2; //difference between ref_bkg_original and ref_bkg calculated
	delta_avg = 2;  //difference between delta average and and ref_bkg and ref_mat calculated
	n_it = 0 ; 
	
	//Calculating X and Y.
	X = equalX(mat_val, bkg_val, ref_mat, ref_bkg);
	Y = equalY(mat_val, bkg_val, ref_mat, ref_bkg, X);
	//print("X " + X + " it : " + n_it);
	//print("Y " + Y + " it : " + n_it);
	X1 = X;
	Y1=Y;

	//3.2 While loop:
	// while ( abs(delta_mat) > delta  ||  abs(delta_bkg) > delta  &&  n_it < 5){
	while ( delta_avg > delta  &&  n_it < 5){
		
		selectImage(Target_ID);
		run("Duplicate...",  "Slice_" + slice + " duplicate range=" + slice + "-" + slice);
		Slice_ID = getImageID();
		selectImage(Slice_ID);
		//print("Iteration number " + (n_it+1) + " started......");
		run("Brightness/Contrast...");
		setMinAndMax(X1, Y1);
		run("8-bit");
		
		//3.2.1 Checking the material:
		selectImage(Slice_ID);
		roiManager("Select", 0);
		getHistogram(values,counts,nBins);
//Array.show("title", values, counts);
		
		//Asingning Variables:
		counts_mat_8b = counts;
		values_mat_8b = values;
		max_mat_8b = 0;
		for (i = 0; i < counts_mat_8b.length;i++){
			if(counts_mat_8b[i] > max_mat_8b){
			max_mat_8b = counts_mat_8b[i];
			}
		}
		treshold_mat_8b = max_mat_8b*t_mat;
		counter_mat_8b = 0;
		sum_mat_8b = 0;
		for(i = 0; i < counts_mat_8b.length; i++){
			if(counts_mat_8b[i] > treshold_mat_8b){
			counter_mat_8b = counter_mat_8b + counts_mat_8b[i];
			sum_mat_8b = sum_mat_8b + counts_mat_8b[i]*values_mat_8b[i];	
			}
		}
		mat_val_8b = sum_mat_8b / counter_mat_8b ;
		//print("8bit material average:\n" + mat_val_8b);
		
		//3.2.2.Checking the background:
		roiManager("Select", 1);
		nBins = 256;
        		getHistogram(values,counts,nBins);
//Array.show("title", values, counts);	
		
		//Asingning Variables:
		counts_bkg_8b = counts;
		values_bkg_8b = values;
		max_bkg_8b = 0;
		for (i = 0; i < counts_bkg_8b.length;i++){
			if(counts_bkg_8b[i] > max_bkg_8b){
			max_bkg_8b = counts_bkg_8b[i];
			}
		}
		treshold_bkg_8b = max_bkg_8b*t_bkg;
		//Calculating the average using the treshold:
		counter_bkg_8b = 0;
		sum_bkg_8b = 0;
		for(i = 0; i < counts_bkg_8b.length; i++){
			if(counts_bkg_8b[i] > treshold_bkg_8b){
			counter_bkg_8b = counter_bkg_8b + counts_bkg_8b[i];
			sum_bkg_8b = sum_bkg_8b + counts_bkg_8b[i]*values_bkg_8b[i];	
			}
		}
		bkg_val_8b = sum_bkg_8b / counter_bkg_8b ;
		//print("8bit background average:\n" + bkg_val_8b);
		
		//Deltas:
		delta_mat = ref_mat - mat_val_8b;
		delta_bkg = ref_bkg - bkg_val_8b;
		delta_avg = (abs(delta_mat) + abs(delta_bkg) )/2;
			
		/*print("bkg_val_8b = " + bkg_val_8b);
		print("mat_val_8b = " + mat_val_8b);
		print("X = " + X);
		print("Y = " + Y);
		print("delta_bkg = " + delta_bkg);
		print("delta_mat = " + delta_mat);*/
		n_it++;
		////////////Changing the X1, Y1 values

		if (n_it > 5){
			print("slice number:" + slice + "8bit background average:" + bkg_val_8 + "8bit mat average:\n" + mat_val_8b);
			print("break");
        			break;
		
		}


		
		X1 = round(X1 - 0.25*delta_bkg*65535/255);
		Y1 = round(Y1 - 0.25*delta_mat*65535/255);
		selectImage(Slice_ID);
		close();
	}	

	//4.0 Creating the new stack:
	//4.1 Duplicating:
	selectImage(Target_ID);
	run("Duplicate...",  "Slice_" + slice + " duplicate range=" + slice + "-" + slice);
	Slice_ID = getImageID();
	Slice_Title = getTitle();
	selectImage(Slice_ID);
	//4.2 Apling the changes:
	run("Brightness/Contrast...");
	setMinAndMax(X, Y);
	run("8-bit");
	//4.3 Concatenating:
	run("Concatenate...", "  title=[Eq_Stack] image1=" + New_Stack +" image2=" + Slice_Title + " image3=[-- None --]");
	New_Stack_ID = getImageID();
	
	if (slice == start_slice){
		X_s = X;
		Y_s = Y;
	}
	if (slice == end_slice){
		X_f = X;
		Y_f = Y;
	}
	
	if (!fix_bkg_ROI){
		roiManager("Select", 1);
		roiManager("Delete");
	}
	if (!fix_mat_ROI){
		roiManager("Select", 0);
		roiManager("Delete");
	}
	//if (!fix_mat_ROI  &&  !fix_bkg_ROI) {
	//	roiManager("Deselect");
	//	roiManager("Delete");
	//}	
}

//5.Apllying the changes to the rest of the volume

selectImage(New_Stack_ID);
run("Set Slice...", "slice=" + 1);
run("Delete Slice");
selectImage(Target_ID);

for (slice=(end_slice+1); slice <= Target_Slices; slice++) {
	print("\\Clear");
	print("Progress : " + (((slice-end_slice)/Target_Slices*100) + (end_slice-start_slice)/Target_Slices*100) + "%" );
	selectImage(Target_ID);
	run("Set Slice...", "slice=" + slice);
	//4.1 Duplicating:
	run("Duplicate...",  "Slice_" + slice + " duplicate range=" + slice + "-" + slice);
	Slice_ID = getImageID();
	Slice_Title = getTitle();
	selectImage(Slice_ID);
	//4.2 Apling the changes:
	run("Brightness/Contrast...");
	setMinAndMax(X_f, Y_f);
	run("8-bit");
	//4.3 Concatenating:
	run("Concatenate...", "  title=[Eq_Stack] image1=" + New_Stack +" image2=" + Slice_Title + " image3=[-- None --]");
}

for (slice=(start_slice-1); slice > 0; slice--) {
	print("\\Clear");
	print("Progress : " + (((start_slice-slice)/Target_Slices*100) + (Target_Slices-start_slice)/Target_Slices*100)+ "%" );
	selectImage(Target_ID);
	run("Set Slice...", "slice=" + slice);
	//4.1 Duplicating:
	run("Duplicate...",  "Slice_" + slice + " duplicate range=" + slice + "-" + slice);
	Slice_ID = getImageID();
	Slice_Title = getTitle();
	selectImage(Slice_ID);
	//4.2 Apling the changes:
	run("Brightness/Contrast...");
	setMinAndMax(X_s, Y_s);
	run("8-bit");
	Slice_ID = getImageID();
	Slice_Title = getTitle();
	//4.3 Concatenating:
	run("Concatenate...", "  title=[Eq_Stack] image1=" + Slice_Title +" image2=" + New_Stack + " image3=[-- None --]");
	New_Stack_ID = getImageID();
}
print("\\Clear");
print("Progress : 100%" );

//5.0 Saving the volume:
Result_Name = replace(Target_Title, "16b","EQ4_2_8b");
selectImage(New_Stack_ID);
saveAs("Raw Data", Target_Directory + Result_Name +".raw");
print("Results saved as: " + Result_Name + ".raw" + " in " + Target_Directory);

//5.1 Closing all:
selectImage(Target_ID);
close();
//selectWindow("ROI Manager") ;
//run ("Close") ;
selectWindow("B&C") ;
run ("Close") ;

//5.1 Finish message:

print("The stack was analized from slice " + start_slice + " to slice " + end_slice + ".");

print("Threshold material = " + t_mat);
print("Threshold background = " + t_bkg);
print("Reference material = " + ref_mat_original);
print("Reference background = " + ref_bkg_original);
print("Tolerance = " + delta);

Dialog.create("Equalization done!");
Dialog.addMessage("Equalization Completed:");
Dialog.addMessage("Results saved as: " + Result_Name + ".raw");
Dialog.addMessage("Folder: " + Target_Directory);
Dialog.show();





