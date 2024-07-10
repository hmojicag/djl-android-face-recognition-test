package io.github.hmojicag;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.net.Uri;
import android.os.Environment;
import android.text.TextUtils;
import android.util.Log;
import com.google.android.gms.tasks.Task;
import com.google.android.gms.tasks.Tasks;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
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
    private static final String CACHED_BIOPHOTOS_JSON = "biophotos.json";
    private static final String CACHED_ATTPHOTOS_JSON = "attphotos.json";
    private static final float featureThresholdForMatch = 0.70f;
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
            List<BioPhoto> allBioPhotos = loadAllBioPhotos();
            List<AttPhoto> allAttPhotos = loadAllAttPhotos();
            Log.i(TAG, "Finished loading photos");
            List<String> goodMatch = new ArrayList<>();
            List<String> mismatch = new ArrayList<>();
            List<String> noMatch = new ArrayList<>();
            List<String> ignoreNoMatch = new ArrayList<>();
            for(AttPhoto attPhoto : allAttPhotos) {
                BioPhoto bioPhoto = getBioPhotoMatch(attPhoto, allBioPhotos);
                if (bioPhoto == null) {
                    // No match? Check if there exist a BioPhoto for that employee
                    boolean bpExist = allBioPhotos.stream().anyMatch(bp ->
                            bp.getEmpresa() == attPhoto.getEmpresa() && bp.getEmpleado() == attPhoto.getEmpleado());
                    if (bpExist) {
                        noMatch.add(String.format("No Match %s", attPhoto));
                    } else {
                        ignoreNoMatch.add(String.format("No BioPhoto for %s", attPhoto));
                    }
                } else if(attPhoto.getEmpresa() == bioPhoto.getEmpresa() && attPhoto.getEmpleado() == bioPhoto.getEmpleado()) {
                    goodMatch.add(String.format("Good %s %s", bioPhoto, attPhoto));
                } else {
                    mismatch.add(String.format("Mismatch %s %s", bioPhoto, attPhoto));
                }
            }
            String goodMatchStr = TextUtils.join("\n", goodMatch);
            String mismatchStr = TextUtils.join("\n", mismatch);
            String noMatchStr = TextUtils.join("\n", noMatch);
            String ignoreNoMatchStr = TextUtils.join("\n", ignoreNoMatch);
            float totalAttPhotos = allAttPhotos.size();
            float goodMatches = goodMatch.size();
            float pSuccess = totalAttPhotos - goodMatches / totalAttPhotos;
            Log.i(TAG, "Complete.");
        });
    }

    private List<BioPhoto> loadAllBioPhotos() {
        // Ex. "/storage/emulated/0/Download/BioPhotos/1-1430.jpg"
        Path pathBioPhotosDir = Paths.get(Environment.getExternalStorageDirectory().toString(), "Download", "BioPhotos");
        File bioPhotosDir = new File(pathBioPhotosDir.toString());
        File[] bioPhotoFiles = bioPhotosDir.listFiles();
        List<BioPhoto> bioPhotos = new ArrayList<>();
        for(File bioPhotoFile : bioPhotoFiles) {
            // Ex. "file:///storage/emulated/0/Download/BioPhotos/1-1430.jpg"
            String fileUriPath = String.format("file://%s", bioPhotoFile.getPath());
            BioPhoto bioPhoto = getBioPhoto(fileUriPath);
            if (bioPhoto != null) {
                bioPhotos.add(bioPhoto);
            }
        }
        return bioPhotos;
    }

    private List<BioPhoto> loadAllBioPhotosFromJson() {
        try {
            Path bioPhotosPath = Paths.get(context.getFilesDir().getPath(), CACHED_BIOPHOTOS_JSON);
            if (Files.exists(bioPhotosPath)) {
                Gson gson = new Gson();
                Type listType = new TypeToken<ArrayList<BioPhoto>>(){}.getType();
                String bioPhotosStr = new String(Files.readAllBytes(bioPhotosPath));
                return gson.fromJson(bioPhotosStr, listType);
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }

    private void saveBioPhotosList(List<BioPhoto> bioPhotos) {
        try {
            Path bioPhotosPath = Paths.get(context.getFilesDir().getPath(), CACHED_BIOPHOTOS_JSON);
            if (!Files.exists(bioPhotosPath)) {
                Gson gson = new Gson();
                byte[] jsonBioPhotos = gson.toJson(bioPhotos).getBytes();
                Files.write(bioPhotosPath, jsonBioPhotos);
            }
        } catch (Exception ex) {
            ex.printStackTrace();
        }
    }


    private List<AttPhoto> loadAllAttPhotos() {
        // Ex. "/storage/emulated/0/Download/AttPhotos/52-20240619082918-62157.jpg"
        Path pathAttPhotosDir = Paths.get(Environment.getExternalStorageDirectory().toString(), "Download", "AttPhotos");
        File attPhotosDir = new File(pathAttPhotosDir.toString());
        File[] attPhotoFiles = attPhotosDir.listFiles();
        List<AttPhoto> attPhotos = new ArrayList<>();
        for(File attPhotoFile : attPhotoFiles) {
            // Ex. "file:///storage/emulated/0/Download/AttPhotos/52-20240619082918-62157.jpg"
            String fileUriPath = String.format("file://%s", attPhotoFile.getPath());
            AttPhoto attPhoto = getAttPhoto(fileUriPath);
            if (attPhoto != null) {
                attPhotos.add(attPhoto);
            }
        }
        return attPhotos;
    }

    private BioPhoto getBioPhoto(String fileUriPath) {
        Log.i(TAG, String.format("Getting BioPhoto for %s", fileUriPath));
        FeaturesResult featuresResult = loadPhotoAndExtractFeatures(fileUriPath);
        if (featuresResult != null) {
            try {
                String fileName = Paths.get(fileUriPath).getFileName().toString();
                String[] empresaEmpleado = fileName.split("\\.")[0].split("-");
                int empresa = Integer.parseInt(empresaEmpleado[0]);
                int empleado = Integer.parseInt(empresaEmpleado[1]);
                //return new BioPhoto(empresa, empleado, featuresResult.fullPhoto, featuresResult.croppedPhoto, featuresResult.features);
                return new BioPhoto(empresa, empleado, null, null, featuresResult.features);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return null;
    }

    private AttPhoto getAttPhoto(String fileUriPath) {
        Log.i(TAG, String.format("Getting AttPhoto for %s", fileUriPath));
        FeaturesResult featuresResult = loadPhotoAndExtractFeatures(fileUriPath);
        if (featuresResult != null) {
            try {
                String fileName = Paths.get(fileUriPath).getFileName().toString();
                String[] empresaEmpleado = fileName.split("\\.")[0].split("-");
                int empresa = Integer.parseInt(empresaEmpleado[0]);
                long randomId = Long.parseLong(empresaEmpleado[1]);
                int empleado = Integer.parseInt(empresaEmpleado[2]);
                //return new AttPhoto(empresa, empleado, featuresResult.fullPhoto, featuresResult.croppedPhoto, featuresResult.features, randomId);
                return new AttPhoto(empresa, empleado, null, null, featuresResult.features, randomId);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return null;
    }

    private FeaturesResult loadPhotoAndExtractFeatures(String fileUriPath) {
        PhotosResult photosResult = loadPhoto(fileUriPath);
        if (photosResult != null) {
            try {
                Image image = ImageFactory.getInstance().fromImage(photosResult.croppedPhoto);
                float[] features = predictor.predict(image);
                return new FeaturesResult(photosResult.fullPhoto, photosResult.croppedPhoto, features);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        return null;
    }

    private PhotosResult loadPhoto(String fileUriPath) {
        FacesResult facesResult = getFacesFromInputImage(fileUriPath);
        if (facesResult == null) {
            Log.i(TAG, String.format("There was a problem identifying faces on %s", fileUriPath));
            return null;
        }
        List<Face> faces = facesResult.faces;
        Bitmap fullPhoto = facesResult.fullPhoto;
        if (faces == null || faces.isEmpty()) {
            Log.i(TAG, String.format("No faces identified, skipping %s", fileUriPath));
            return null;
        }
        if (faces.size() > 1) {
            Log.i(TAG, String.format("More than one face identified, skipping %s", fileUriPath));
            return null;
        }
        // Crop the original image to just the face
        try {
            Face face = faces.get(0);
            /**
             * left	    int: The X coordinate of the left side of the rectangle
             * top	    int: The Y coordinate of the top of the rectangle
             * right	int: The X coordinate of the right side of the rectangle
             * bottom	int: The Y coordinate of the bottom of the rectangle
             */
            RectF boundingBox = new RectF(
                    Math.max(face.getBoundingBox().left, 0),
                    Math.max(face.getBoundingBox().top, 0),
                    Math.min(face.getBoundingBox().right, fullPhoto.getWidth()),
                    Math.min(face.getBoundingBox().bottom, fullPhoto.getHeight())
            );
            Bitmap croppedPhoto = Bitmap.createBitmap(fullPhoto,
                    (int) boundingBox.left,
                    (int) boundingBox.top,
                    (int) boundingBox.width(),
                    (int) boundingBox.height());
            return new PhotosResult(fullPhoto, croppedPhoto);
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return null;
    }

    private FacesResult getFacesFromInputImage(String fileUriPath) {
        try {
            InputImage image = InputImage.fromFilePath(context, Uri.parse(fileUriPath));
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
                    .add(new Resize(160))
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

    /**
     * Returns the biophoto that makes match with the sampleBioPhoto passed as parameter.
     * Returns null if sampleBioPhoto is not found in the dataset
     * @param attPhoto The sample photo to compare against the data set
     * @param bioPhotos The dataset
     * @return
     */
    private BioPhoto getBioPhotoMatch(AttPhoto attPhoto, List<BioPhoto> bioPhotos) {
        List<BioPhotoMatch> matchedPhotos = new ArrayList<>();
        for(BioPhoto bioPhoto: bioPhotos) {
            float result = calculateSimilitude(attPhoto.getFeatures(), bioPhoto.getFeatures());
            if (result >= featureThresholdForMatch) {
                matchedPhotos.add(new BioPhotoMatch(result, bioPhoto));
            }
        }

        if (matchedPhotos.isEmpty()) {
            Log.i(TAG, "Not match");
            return null;
        }

        if (matchedPhotos.size() == 1) {
            Log.i(TAG, "Single match");
            return matchedPhotos.get(0).getBioPhoto();
        }

        // Resolve collision
        Log.i(TAG, "Multiple match, resolving collision");
        return matchedPhotos
                .stream()
                .sorted()
                .findFirst()
                .map(matchedPhoto -> matchedPhoto.getBioPhoto())
                .orElse(null);
    }

    private static float calculateSimilitude(float[] feature1, float[] feature2) {
        float ret = 0.0f;
        float mod1 = 0.0f;
        float mod2 = 0.0f;
        int length = feature1.length;
        for (int i = 0; i < length; ++i) {
            ret += feature1[i] * feature2[i];
            mod1 += feature1[i] * feature1[i];
            mod2 += feature2[i] * feature2[i];
        }
        return (float) ((ret / Math.sqrt(mod1) / Math.sqrt(mod2) + 1) / 2.0f);
    }
}
