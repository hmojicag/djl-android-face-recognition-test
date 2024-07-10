package io.github.hmojicag;

public class BioPhotoMatch implements Comparable<BioPhotoMatch> {
    private float similitude;
    private BioPhoto bioPhoto;

    public BioPhotoMatch(float similitude, BioPhoto bioPhoto) {
        this.similitude = similitude;
        this.bioPhoto = bioPhoto;
    }

    public float getSimilitude() {
        return similitude;
    }

    public void setSimilitude(float similitude) {
        this.similitude = similitude;
    }

    public BioPhoto getBioPhoto() {
        return bioPhoto;
    }

    public void setBioPhoto(BioPhoto bioPhoto) {
        this.bioPhoto = bioPhoto;
    }

    @Override
    public int compareTo(BioPhotoMatch o) {
        // Comparator reversed to order in Descending order
        return Float.compare(o.getSimilitude(), this.getSimilitude());
    }
}
