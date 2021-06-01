package com.example.datacollectorapp;

import android.content.Context;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.IOException;

public class TensorFlowClassifier {
    private static final String MODEL_FILE = "model.pb";
    private static final String INPUT_NODE = "input";
    private static final String[] OUTPUT_NODES = {"output"};
    private static final String OUTPUT_NODE = "output";
    private static final long[] INPUT_SIZE = {1, 100, 9};
    private static final int OUTPUT_SIZE = 7;

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private TensorFlowInferenceInterface inferenceInterface;

    public TensorFlowClassifier(Context context) {
        inferenceInterface = new TensorFlowInferenceInterface(context.getAssets(), MODEL_FILE);
    }

    public float[] predictProbabilities(float[] data) {
        float[] result = new float[OUTPUT_SIZE];
        inferenceInterface.feed(INPUT_NODE, data, INPUT_SIZE);
        inferenceInterface.run(OUTPUT_NODES);
        inferenceInterface.fetch(OUTPUT_NODE, result);
        return result;
    }

}
