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

    public int getScore() throws Exception {
         final String arffTrainData = "out.arff";

        AbstractClassifier classifier = trainModel(arffTrainData, 2000000);

        Instance ins = new weka.core.SparseInstance(2000000);

        //TODO
        double vector = 0.0;
        for(int i=0; i<2000000; i++) {
            ins.setValue(i, vector);
        }

        double star = classifier.classifyInstance(ins);
        return (int)star;
    }
}
