package org.genesys.decide;

import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.ArrayList;

/**
 * Created by utcs on 6/5/17.
 */
public class RandomDecider implements Decider {

    @Override
    public String decide(List<String> trail, List<String> candidates) {
        Random rand = new Random();
        int p = rand.nextInt(candidates.size())+1;
        return candidates.get(p-1);
    }

    @Override
    public String decideSketch(List<String> trail, List<String> candidates, int child) { return decide(trail, candidates); }

}
