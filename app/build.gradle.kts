plugins {
    alias(libs.plugins.android.application)
}

android {
    namespace = "io.github.hmojicag"
    compileSdk = 34

    defaultConfig {
        applicationId = "io.github.hmojicag"
        minSdk = 27
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    implementation(libs.appcompat)
    implementation(libs.material)
    implementation(libs.constraintlayout)
    implementation(libs.navigation.fragment)
    implementation(libs.navigation.ui)
    implementation(libs.face.detection)

    // https://djl.ai/android/
    // https://github.com/deepjavalibrary/djl/tree/master/android
    implementation(platform("ai.djl:bom:0.27.0"))
    implementation("ai.djl:api")
    implementation("ai.djl.android:core")
    runtimeOnly("ai.djl.pytorch:pytorch-engine")
    runtimeOnly("ai.djl.android:pytorch-native")
    runtimeOnly("ai.djl.android:onnxruntime")

    testImplementation(libs.junit)
    androidTestImplementation(libs.ext.junit)
    androidTestImplementation(libs.espresso.core)
}