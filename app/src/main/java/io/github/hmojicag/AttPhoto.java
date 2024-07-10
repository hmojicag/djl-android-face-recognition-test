package io.github.hmojicag;

import android.graphics.Bitmap;

public class AttPhoto extends BioPhoto {
    private long id;

    public AttPhoto(int empresa, int empleado, Bitmap fullPhoto, Bitmap croppedFace, float[] features, long id) {
        super(empresa, empleado, fullPhoto, croppedFace, features);
        this.id = id;
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    @Override
    public String toString() {
        return "AttPhoto{" +
                "empresa=" + getEmpresa() +
                ", empleado=" + getEmpleado() +
                ", id=" + id +
                '}';
    }
}
