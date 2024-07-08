package io.github.hmojicag;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.concurrent.Executors;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Pipeline;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class DjlFaceRecognitionTest {
    private static final String TAG = "DjlFaceRecognitionTest";
    private static final String MODEL_ZIP_DOWNLOAD_FILENAME = "face_feature.zip";
    private Context context;
    private FaceDetector faceDetector;
    private Criteria<Image, float[]> criteria;
    private ZooModel<Image, float[]> model;
    private Predictor<Image, float[]> predictor;

    public DjlFaceRecognitionTest(Context context) {
        this.context = context;
        faceDetector = FaceDetection.getClient();
        System.setProperty("ai.djl.offline", "true");
        copyPytorchModelFromAssetsToFileSystem();
        criteria = buildCriteria();
        model = loadZooModel(criteria);
        predictor = createPredictor(model);
    }

    public void runTest() {
        Executors.newSingleThreadExecutor().execute(() -> {
            BioPhoto bioPhotoTest = getBioPhoto("file:///storage/emulated/0/Download/1-1430.jpg");
            Log.i(TAG, "");
        });
    }

    private AttPhoto getAttPhoto(String path) {
        return null;
    }

    private List<BioPhoto> loadAllBioPhotos() {
        Path pathBioPhotosDir = Paths.get(Environment.getExternalStorageDirectory().toString(), "Download", "BioPhotos");
        File bioPhotosDir = new File(pathBioPhotosDir.toString());
        File[] bioPhotoFiles = bioPhotosDir.listFiles();

        return null;
    }

    private BioPhoto getBioPhoto(String path) {
        FeaturesResult featuresResult = loadPhotoAndExtractFeatures(path);
        if (featuresResult != null) {
            try {
                String fileName = Paths.get(path).getFileName().toString();
                String[] empresaEmpleado = fileName.split("\\.")[0].split("-");
                int empresa = Integer.parseInt(empresaEmpleado[0]);
                int empleado = Integer.parseInt(empresaEmpleado[1]);
                return new BioPhoto(empresa, empleado, featuresResult.fullPhoto, featuresResult.croppedPhoto, featuresResult.features);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return null;
    }

    private FeaturesResult loadPhotoAndExtractFeatures(String path) {
        PhotosResult photosResult = loadPhoto(path);
        if (photosResult != null) {
            try {
                Image image = ImageFactory.getInstance().fromImage(photosResult.croppedPhoto);
                float[] features = predictor.predict(image);
                return new FeaturesResult(photosResult.fullPhoto, photosResult.fullPhoto, features);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return null;
    }

    private PhotosResult loadPhoto(String path) {
        FacesResult facesResult = getFacesFromInputImage(path);
        if (facesResult == null) {
            Log.i(TAG, String.format("There was a problem identifying faces on %s", path));
            return null;
        }
        List<Face> faces = facesResult.faces;
        Bitmap fullPhoto = facesResult.fullPhoto;
        if (faces == null) {
            Log.i(TAG, String.format("No faces identified, skipping %s", path));
            return null;
        }
        if (faces.size() > 1) {
            Log.i(TAG, String.format("More than one face identified, skipping %s", path));
            return null;
        }
        // Crop the original image to just the face
        Face face = faces.get(0);
        RectF boundingBox = new RectF(face.getBoundingBox());
        Bitmap croppedPhoto = Bitmap.createBitmap(fullPhoto,
                (int) boundingBox.left,
                (int) boundingBox.top,
                (int) boundingBox.width(),
                (int) boundingBox.height());
        return new PhotosResult(fullPhoto, croppedPhoto);
    }

    private FacesResult getFacesFromInputImage(String path) {
        try {
            InputImage image = InputImage.fromFilePath(context, Uri.parse(path));
            Task<List<Face>> result = faceDetector.process(image);
            Tasks.await(result);
            return new FacesResult(result.getResult(), image.getBitmapInternal());
        } catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    private void copyPytorchModelFromAssetsToFileSystem() {
        try {
            Path modelPath = Paths.get(context.getFilesDir().getPath(), MODEL_ZIP_DOWNLOAD_FILENAME);
            if (!Files.exists(modelPath)) {
                try(InputStream inputStream = context.getAssets().open("pytorch_models/face_feature.zip")) {
                    // Files.copy(inputStream, modelPath, StandardCopyOption.REPLACE_EXISTING);
                    try (FileOutputStream fos = context.openFileOutput(MODEL_ZIP_DOWNLOAD_FILENAME, Context.MODE_PRIVATE)) {
                        byte[] buffer = new byte[8192];
                        int length;
                        while( (length = inputStream.read(buffer)) != -1) {
                            fos.write(buffer, 0, length);
                        }
                    }
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            throw new RuntimeException(ex);
        }
    }

    private Criteria<Image, float[]> buildCriteria() {
        try {
            Log.i(TAG, "Building Criteria face_feature with Engine PyTorch");
            long time = System.currentTimeMillis();
            Criteria<Image, float[]> criteria =
                    Criteria.builder()
                            .setTypes(Image.class, float[].class)
                            .optModelPath(new File(context.getFilesDir(), MODEL_ZIP_DOWNLOAD_FILENAME).toPath())
                            .optModelName("face_feature") // specify model file prefix
                            .optTranslator(new FaceFeatureTranslator())
                            .optProgress(new ProgressBar())
                            .optEngine("PyTorch") // Use PyTorch engine
                            .build();
            Log.i(TAG, "Loading criteria took " + (System.currentTimeMillis()-time) + " ms");
            return criteria;
        } catch (Exception ex) {
            ex.printStackTrace();
            throw new RuntimeException(ex);
        }
    }

    private ZooModel<Image, float[]> loadZooModel(Criteria<Image, float[]> criteria) {
        try {
            Log.i(TAG, "Loading ZooModel");
            long time = System.currentTimeMillis();
            ZooModel<Image, float[]> model = criteria.loadModel();
            Log.i(TAG, "Loading ZooModel took " + (System.currentTimeMillis()-time) + " ms");
            return model;
        } catch (Exception ex) {
            ex.printStackTrace();
            throw new RuntimeException(ex);
        }
    }

    private Predictor<Image, float[]> createPredictor(ZooModel<Image, float[]> model) {
        Log.i(TAG, "Create Predictor");
        long time = System.currentTimeMillis();
        Predictor<Image, float[]> predictor = model.newPredictor();
        Log.i(TAG, "Create Predictor took " + (System.currentTimeMillis()-time) + " ms");
        return predictor;
    }

    class FacesResult {
        List<Face> faces;
        Bitmap fullPhoto;
        public FacesResult(List<Face> faces, Bitmap fullPhoto) {
            this.faces = faces;
            this.fullPhoto = fullPhoto;
        }
    }

    class PhotosResult {
        Bitmap fullPhoto;
        Bitmap croppedPhoto;
        public PhotosResult(Bitmap fullPhoto, Bitmap croppedPhoto) {
            this.fullPhoto = fullPhoto;
            this.croppedPhoto = croppedPhoto;
        }
    }

    class FeaturesResult extends PhotosResult {
        float[] features;
        public FeaturesResult(Bitmap fullPhoto, Bitmap croppedPhoto, float[] features) {
            super(fullPhoto, croppedPhoto);
            this.features = features;
        }
    }

    private final class FaceFeatureTranslator implements Translator<Image, float[]> {

        FaceFeatureTranslator() {}

        /** {@inheritDoc} */
        @Override
        public NDList processInput(TranslatorContext ctx, Image input) {
            NDArray array = input.toNDArray(ctx.getNDManager(), Image.Flag.COLOR);
            Pipeline pipeline = new Pipeline();
            pipeline
                    // .add(new Resize(160))
                    .add(new ToTensor())
                    .add(
                            new Normalize(
                                    new float[] {127.5f / 255.0f, 127.5f / 255.0f, 127.5f / 255.0f},
                                    new float[] {
                                            128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f
                                    }));

            return pipeline.transform(new NDList(array));
        }

        /** {@inheritDoc} */
        @Override
        public float[] processOutput(TranslatorContext ctx, NDList list) {
            NDList result = new NDList();
            long numOutputs = list.singletonOrThrow().getShape().get(0);
            for (int i = 0; i < numOutputs; i++) {
                result.add(list.singletonOrThrow().get(i));
            }
            float[][] embeddings =
                    result.stream().map(NDArray::toFloatArray).toArray(float[][]::new);
            float[] feature = new float[embeddings.length];
            for (int i = 0; i < embeddings.length; i++) {
                feature[i] = embeddings[i][0];
            }
            return feature;
        }
    }
}
