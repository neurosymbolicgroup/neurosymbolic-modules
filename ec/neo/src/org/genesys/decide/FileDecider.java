package org.genesys.decide;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

/**
 * Created by utcs on 9/19/17.
 */
public class FileDecider implements Decider {

    private int program_ = 0;
    private int child_ = 0;
    private List<List<String>> sketches_ = new ArrayList<>();

    public FileDecider(String filename){

        BufferedReader br = null;
        try {
            br = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        try {
            StringBuilder sb = new StringBuilder();
            String line = null;
            try {
                line = br.readLine();
            } catch (IOException e) {
                e.printStackTrace();
            }

            while (line != null) {
                String program = "";
                ArrayList<String> p = new ArrayList<>();
                if (line.startsWith("l(a,b).(< a b)")){
                    program = line.substring(15,line.length());
                    String [] parts = program.split(" ");
                    p.add("l(a,b).(< a b)");
                    for (int i = 0; i < parts.length; i++) {
                        p.add(parts[i]);
                    }
                    sketches_.add(p);
                } else if (line.startsWith("l(a,b).(== a b)")){
                    program = line.substring(16,line.length());
                    String [] parts = program.split(" ");
                    p.add("l(a,b).(== a b)");
                    for (int i = 0; i < parts.length; i++) {
                        p.add(parts[i]);
                    }
                    sketches_.add(p);
                } else if (line.startsWith("l(a,b).(> a b)")){
                    program = line.substring(15,line.length());
                    String [] parts = program.split(" ");
                    p.add("l(a,b).(> a b)");
                    for (int i = 0; i < parts.length; i++) {
                        p.add(parts[i]);
                    }
                    sketches_.add(p);
                } else if (line.startsWith("l(a,b).(/ a b)")){
                    program = line.substring(15,line.length());
                    String [] parts = program.split(" ");
                    p.add("l(a,b).(/ a b)");
                    for (int i = 0; i < parts.length; i++) {
                        p.add(parts[i]);
                    }
                    sketches_.add(p);
                } else {
                    program = line;
                    String [] parts = program.split(" ");
                    for (int i = 0; i < parts.length; i++) {
                        p.add(parts[i]);
                    }
                    sketches_.add(p);
                }

                try {
                    line = br.readLine();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            String everything = sb.toString();
        } finally {
            try {
                br.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

    }

    @Override
    public String decideSketch(List<String> trail, List<String> candidates, int child) {
        if (program_ >= sketches_.size())
            return null;
        String decision = sketches_.get(program_).get(child);
        if (!candidates.contains(decision)) {
            decision = "";
            program_++;
        }

        return decision;
    }

    @Override
    public String decide(List<String> trail, List<String> candidates) {
        return candidates.get(0);
    }

}
