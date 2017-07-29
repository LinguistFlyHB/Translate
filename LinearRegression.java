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

    public static AbstractClassifier trainModel(String arffFile, int classIndex) throws Exception {

        File inputFile = new File(arffFile); //训练文件
        ArffLoader loader = new ArffLoader();
        loader.setFile(inputFile);
        Instances insTrain = loader.getDataSet(); // 读入训练文件
        insTrain.setClassIndex(classIndex);

        LinearRegression linear = new LinearRegression();
        linear.buildClassifier(insTrain);//根据训练数据构造分类器

        return linear;
    }

    public static void main(String[] args) throws Exception {

        final String arffTrainData = "houses.arff";

        AbstractClassifier classifier = trainModel(arffTrainData, 5);

        Instance ins = new weka.core.SparseInstance(5);
        ins.setValue(0, 990.8);
        ins.setValue(1, 1080.8);
        ins.setValue(2, 3);
        ins.setValue(3, 0);
        ins.setValue(4, 1);

        double price = classifier.classifyInstance(ins);
        System.out.println("Price: " + price);
    }
}