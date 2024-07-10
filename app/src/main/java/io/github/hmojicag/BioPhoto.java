package io.github.hmojicag;

import android.graphics.Bitmap;

public class BioPhoto {

    private int empresa;

    private int empleado;

    private Bitmap fullPhoto;

    private Bitmap croppedFace;

    // Features extracted using Facenet Pytorch model.
    private float[] features;

    public BioPhoto(int empresa, int empleado, Bitmap fullPhoto, Bitmap croppedFace, float[] features) {
        this.empresa = empresa;
        this.empleado = empleado;
        this.fullPhoto = fullPhoto;
        this.croppedFace = croppedFace;
        this.features = features;
    }

    public int getEmpresa() {
        return empresa;
    }

    public void setEmpresa(int empresa) {
        this.empresa = empresa;
    }

    public int getEmpleado() {
        return empleado;
    }

    public void setEmpleado(int empleado) {
        this.empleado = empleado;
    }

    public Bitmap getFullPhoto() {
        return fullPhoto;
    }

    public void setFullPhoto(Bitmap fullPhoto) {
        this.fullPhoto = fullPhoto;
    }

    public Bitmap getCroppedFace() {
        return croppedFace;
    }

    public void setCroppedFace(Bitmap croppedFace) {
        this.croppedFace = croppedFace;
    }

    public float[] getFeatures() {
        return features;
    }

    public void setFeatures(float[] features) {
        this.features = features;
    }

    @Override
    public String toString() {
        return "BioPhoto{" +
                "empresa=" + empresa +
                ", empleado=" + empleado +
                '}';
    }
}
