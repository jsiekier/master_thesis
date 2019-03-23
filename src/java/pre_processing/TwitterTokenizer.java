import cmu.arktweetnlp.Tagger;
import org.tartarus.snowball.ext.PorterStemmer;
import java.io.*;
import java.util.*;

public class TwitterTokenizer {
    private Tagger tag;


    public TwitterTokenizer(String modelPath) throws IOException {
        this.tag = new Tagger();
        this.tag.loadModel(modelPath);//model.20120919"
    }

    public void countWordsCSV(String outPath,String csvPath) throws IOException{

        try {
            Map<String,Long> counting = new HashMap<>();
            long all = 0;
            BufferedReader br = new BufferedReader(new FileReader(csvPath));
            String line=null;
            while ((line = br.readLine()) != null) {

                String[] entry = line.split("\t");

                all+=count(entry[4],counting);

            }
            writeSortedWords(outPath, all, counting);
            br.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

    }

    private void writeSortedWords(String outPath,long all,Map<String,Long>counting) throws IOException {
        FileWriter writer = new FileWriter(outPath);
        writer.write(all+"\n");

        Comparator<String> comparator = new ValueComparator(counting);
        TreeMap<String, Long> sorted = new TreeMap<>(comparator);
        sorted.putAll(counting);

        for(Map.Entry<String,Long> entry : sorted.entrySet()) {
            String key = entry.getKey();
            Long value = entry.getValue();
            writer.write(key+"\t"+value+"\t"+(double)value/(double)all+"\n");
        }
        writer.flush();
        writer.close();
    }
    private long count(String text, Map<String, Long> counting) {
        long all=0;
        List<Tagger.TaggedToken> tokens = tag.tokenizeAndTag(text);
        for(Tagger.TaggedToken token: tokens){
            String lowerToken=token.token.toLowerCase();
            if(!(token.tag.equals(",")||token.tag.equals("G")||token.tag.equals("$"))){
                if(token.tag.equals("#")||token.token.contains("#"))
                    lowerToken=lowerToken.substring(1);
                else if(token.tag.equals("@"))
                    lowerToken="<@mention>";
                else if(token.tag.equals("U"))
                    lowerToken="<urlOrMail>";
                else if(token.tag.equals("E"))
                    lowerToken="<emoticon>";

                if(counting.containsKey(lowerToken)){
                    counting.put(lowerToken,counting.get(lowerToken)+1);
                }
                else{
                    counting.put(lowerToken,1L);
                }
                all++;
            }
        }
        return all;
    }


    public Set<String> tokenizeFilterWordsCSV( String wordPath
            ,int threshold,Set<String> stopWords, String outVocPath) throws IOException {

        return buildVocabularyWithOutput(wordPath,threshold,stopWords,outVocPath);
    }

    public void tokenizeFilterWordsCSVWords(String outPath, String csvPath,Set<String> words){
        try {

            BufferedReader br = new BufferedReader(new FileReader(csvPath));
            String row=null;
            FileWriter writer = new FileWriter(outPath);
            while ((row = br.readLine()) != null) {

                String[] entry = row.split("\t");
                String old_text= entry[1];
                String text = entry[4];
                String id = entry[2];
                String label = entry[3];
                List<Tagger.TaggedToken> tokens = tag.tokenizeAndTag(text);

                StringBuilder line = new StringBuilder();
                line.append(old_text).append("\t").append(id).append("\t").append(label).append("\t");
                for (Tagger.TaggedToken token : tokens) {
                    String lowerToken = token.token.toLowerCase();

                    if (token.tag.equals("#") || token.token.contains("#")) {
                        lowerToken = lowerToken.substring(1);
                    } else if (token.tag.equals("@")) {
                        line.append("<@mention> ");
                        inside.append("<@mention> ");
                    } else if (token.tag.equals("U")) {
                        line.append("<urlOrMail> ");
                        inside.append("<urlOrMail> ");
                    } else if (token.tag.equals("E")) {
                        line.append("<emoticon> ");
                        inside.append("<emoticon> ");
                    }

                    if (words.contains(lowerToken) && !(token.tag.equals(",") || token.tag.equals("G")
                            || token.tag.equals("$")
                            || token.tag.equals("@") || token.tag.equals("U") || token.tag.equals("E"))) {

                        line.append(lowerToken);
                        line.append(" ");

                    }
                }

                line.append("\n");
                writer.write(line.toString());
            }
            writer.close();

        }catch (IOException e){
            e.printStackTrace();
        }

    }

    private Set<String> buildVocabularyWithOutput( String wordPath,
                                        int threshold,Set<String> stopWords,String outPath) throws IOException {

        Set<String> keys= new HashSet<>();
        Set<String> words = new HashSet<>();
        Set<String> result = new HashSet<>();
        FileWriter writer = new FileWriter(outPath);
        for (String key : keys) {
            words.addAll(Arrays.asList(key.split(" ")));
        }
        BufferedReader br = new BufferedReader(new FileReader(wordPath));
        String dataLine;
        br.readLine();//skip first counting

        Map<String,Long> input = new HashMap<>();
        while ((dataLine = br.readLine()) != null) {
            String[] data = dataLine.split("\t");
            if (!stopWords.contains(data[0]))// filter stop words
                input.put(data[0],Long.parseLong(data[1]));
        }
        Map<String,Long> counting = new HashMap<>();
        for(String word : words) {
            if(input.containsKey(word)){
                counting.put(word,input.get(word));
            }
        }

        Comparator<String> comparator = new ValueComparator(input);
        TreeMap<String, Long> sorted = new TreeMap<>(comparator);
        sorted.putAll(input);

        int i=0;
        for(Map.Entry<String,Long> entry : sorted.entrySet()) {
            String key = entry.getKey();
            Long value = entry.getValue();
            counting.put(key,value);
            if(i>threshold)
                break;
            i++;
        }

        Comparator<String> comparator2 = new ValueComparator(counting);
        TreeMap<String, Long> sorted2 = new TreeMap<>(comparator2);
        sorted2.putAll(counting);

        Long all=0L;

        for(Map.Entry<String,Long> entry : sorted2.entrySet()) {
            if(entry.getKey().trim().length()>1||entry.getKey().toLowerCase().equals("i")){
            all+= entry.getValue();
            result.add(entry.getKey());}
        }
        writer.write(all+"\n");
        for(Map.Entry<String,Long> entry : sorted2.entrySet()) {
            writer.write(entry.getKey()+"\t"+entry.getValue()+"\t"+(double)entry.getValue()/(double)all+"\n");
        }
        writer.flush();
        writer.close();
        return  result;
    }

    private Map<String,Double> readCounts(String wordPath) throws IOException {
        Map<String,Double> result = new HashMap<>();
        BufferedReader br = new BufferedReader(new FileReader(wordPath));
        String dataLine;
        br.readLine();//skip first counting
        while ((dataLine = br.readLine()) != null) {
            String[] data = dataLine.split("\t");
            result.put(data[0],Double.parseDouble(data[2]));
        }

        return result;
    }

    public void stemm_text(String csvPath_in, String csvPath_out) {
        try {
            PorterStemmer ps = new PorterStemmer();
            FileWriter writer = new FileWriter(csvPath_out);

            if(!csvPath_in.equals("")) {
                BufferedReader br = new BufferedReader(new FileReader(csvPath_in));

                String linee = null;
                linee = br.readLine();
                while ((linee = br.readLine()) != null) {

                    String[] entry = linee.split("\t");
                    if(entry.length>2){
                        String text = entry[3];
                        String id = entry[2];
                        String label = entry[0];

                        List<Tagger.TaggedToken> tokens = tag.tokenizeAndTag(text);
                        boolean take_line=true;
                        for (Tagger.TaggedToken token : tokens) {
                            if (token.tag.equals("U"))
                                take_line=false;
                        }

                        StringBuilder line = new StringBuilder();
                        line.append("\t").append(text).append("\t").append(label).append("\t").append(id).append("\t");
                        for (Tagger.TaggedToken token : tokens) {
                            String lowerToken = token.token.toLowerCase();
                            if (!(token.tag.equals(",") || token.tag.equals("G") || token.tag.equals("$"))) {
                                line.append(" ");
                                if (token.tag.equals("#") || token.token.contains("#"))
                                    line.append(ps.stem(lowerToken.replace("#","")));
                                else if (token.tag.equals("@"))
                                    line.append(token.token);
                                else if (token.tag.equals("U"))
                                    line.append(token.token);
                                else if (token.tag.equals("E"))
                                    line.append(token.token);
                                else
                                    line.append(ps.stem(lowerToken));
                            }
                        }

                        line.append("\n");
                        writer.write(line.toString());
                }}
            }

            writer.close();

        } catch (FileNotFoundException e1) {
            e1.printStackTrace();

        } catch (IOException e1) {
            e1.printStackTrace();
        }

    }

    public void prepareInput(Set<String> words, String in_path, String out_path_x, String out_path_y,String vocab_path) {
        try {
            FileWriter writer_vocab = new FileWriter(vocab_path);

            HashMap<String,Integer> vocab= new HashMap<String,Integer>();
            int i = -1;
            for(String word : words){
                i++;
                vocab.put(word,i);
                writer_vocab.write(word+"\n");
            }
            writer_vocab.close();

            BufferedReader br = new BufferedReader(new FileReader(in_path));
            FileWriter writer_y = new FileWriter(out_path_y);
            FileWriter writer_x = new FileWriter(out_path_x);
            String linee=null;
            Map<String,Integer> id_texts= new TreeMap<>();

            while ((linee = br.readLine()) != null) {
                String[] entry = linee.split("\t");

                if(entry.length==4&&entry[3].split(" ").length>2) {
                    String text = entry[3];
                    String label = entry[2];

                    StringBuilder line_x = new StringBuilder("1 ");

                    String[] splitt_text = text.split(" ");
                    HashMap<String, Integer> help = new HashMap<String, Integer>();
                    for (String word : splitt_text) {
                        if (help.containsKey(word)) {
                            help.put(word, help.get(word) + 1);
                        } else {
                            help.put(word, 1);
                        }
                    }
                    StringBuilder stemm_reP= new StringBuilder();
                    for (String word : help.keySet()) {
                        if(vocab.containsKey(word)) {
                            line_x.append(vocab.get(word) + 1).append(":").append(help.get(word)).append(" ");
                            stemm_reP.append(vocab.get(word)).append(" ");
                        }else{
                            System.out.println("ERROR: "+word+"not insde");
                        }
                    }
                    if(!id_texts.containsKey(stemm_reP.toString())&&help.size()>0){
                        id_texts.put(stemm_reP.toString(),1);
                        writer_x.write(line_x.toString() + "\n");
                        writer_y.write(label.trim() + "\n");

                    }
                }

            }
            writer_x.close();
            writer_y.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
