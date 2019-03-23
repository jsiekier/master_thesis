import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class TokenizerMain {
    // set input file:
    private static String csvPath_in = "input.csv";
    // set model path:
    private static String model_path="model.20120919";
    //output:
    private static String out_path_x= "x.txt";
    private static String out_path_y= "y.txt";
    private static String voc_path="vocab.txt";

    // not important:
    private static String vocabularyPath = "counts_of_all_words.txt";
    private static String csvPath = "stemmed.csv";
    private static String in_path="patrial_output2.csv";

    public static void main(String... args){
        try {


            Set<String> stopWords = new HashSet<>( Arrays.asList("a", "an", "and", "are", "as", "at", "be", "by",
                    "for", "if", "in", "into", "is", "it", "of", "on", "or", "such", "that", "the", "their",
                    "then", "there", "these", "they", "this", "to", "was", "will", "with"));

            TwitterTokenizer t = new TwitterTokenizer(model_path);

            t.stemm_text(csvPath_in,csvPath);
            t.countWordsCSV(vocabularyPath,csvPath);
            Set<String> words=t.tokenizeFilterWordsCSV(vocabularyPath,2449,stopWords,"BOW_counts.csv");
            t.tokenizeFilterWordsCSVWords(in_path,csvPath,words);
            t.prepareInput(words,in_path,out_path_x,out_path_y,voc_path);

        }  catch (IOException e) {
            e.printStackTrace();
        }

    }
}
