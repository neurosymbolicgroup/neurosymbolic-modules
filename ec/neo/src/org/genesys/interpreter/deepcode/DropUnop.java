package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.Unop;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 9/7/17.
 * <p>
 * Given an integer n and array xs, returns the array with the first n elements dropped. (If the
 * length of xs was no larger than n in the first place, an empty array is returned.)
 */
public class DropUnop implements Unop {

    public Object apply(Object obj) {
        List pair = (List) obj;
        assert pair.size() == 2 : pair;
        if (!(pair.get(0) instanceof List) || !(pair.get(1) instanceof Integer))
            return new ArrayList<>();
        assert pair.get(0) instanceof List;
        assert pair.get(1) instanceof Integer;
        List xs = (List) pair.get(0);
        int n = (Integer) pair.get(1);
        List res = new ArrayList();
        int len = xs.size();
        if (len <= n || n < 0)
            return 256;
        else
            res = xs.subList(n, len);
        return res;
    }

    public String toString() {
        return "DROP";
    }
}
