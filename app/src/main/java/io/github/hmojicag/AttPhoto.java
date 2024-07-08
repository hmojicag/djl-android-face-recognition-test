package io.github.hmojicag;

import android.graphics.Bitmap;

public class AttPhoto extends BioPhoto {
    private String id;

    public AttPhoto(int empresa, int empleado, Bitmap fullPhoto, Bitmap croppedFace, float[] features, String id) {
        super(empresa, empleado, fullPhoto, croppedFace, features);
        this.id = id;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }
}
