import weka.classifiers.AbstractClassifier;
import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

import java.io.File;

/**
 * Created by Thpffcj on 2017/7/29.
 */
public class LinearRegression {

    private static AbstractClassifier classifier;

    private static AbstractClassifier trainModel(String arffFile, int classIndex) throws Exception {

        File inputFile = new File(arffFile); //训练文件
        ArffLoader loader = new ArffLoader();
        loader.setFile(inputFile);
        Instances insTrain = loader.getDataSet(); // 读入训练文件
        insTrain.setClassIndex(classIndex);

        LinearRegression linear = new LinearRegression();
        linear.buildClassifier(insTrain);//根据训练数据构造分类器

        return linear;
    }

    public int getScore() throws Exception {

        Instance ins = new weka.core.SparseInstance(40000);

        //TODO
        double vector = 0.0;
        for(int i=0; i<40000; i++) {
            ins.setValue(i, vector);
        }

        double star = classifier.classifyInstance(ins);
        return (int)star;
    }

    public void train() throws Exception {

        final String arffTrainData = "houses.arff";
        classifier = trainModel(arffTrainData, 40000);
    }
}
