package org.genesys.interpreter.deepcode;

import org.genesys.interpreter.Binop;
import org.genesys.interpreter.Unop;
import org.genesys.models.Pair;
import org.genesys.type.AbstractList;
import org.genesys.type.Cons;
import org.genesys.type.EmptyList;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by yufeng on 5/31/17.
 */
public class ZipWith implements Unop {
    private final Binop binop;

    public ZipWith(Binop binop) {
        this.binop = binop;
    }

    public Object apply(Object obj) {
        assert obj != null;
        List<List> pair = (List<List>) obj;
        assert pair.size() == 2 : pair;
        List input1 = pair.get(0);
        List input2 = pair.get(1);
        List targetList = new ArrayList();
        if (input1.size() != input2.size())
            return new ArrayList<>();
        //int min = Math.min(input1.size(), input2.size());
//
//        if (input1.size() != input2.size()) {
//            System.out.println("intput1 = " + input1);
//            System.out.println("intput2 = " + input2);
//            throw new UnsupportedOperationException("Size of inputs need to be equal.");
//        } else {
            for (int i = 0; i < input1.size(); i++) {
                Object elem0 = input1.get(i);
                Object elem1 = input2.get(i);
                Object val = binop.apply(elem0, elem1);
                targetList.add(val);

            }
            return targetList;
        //}
    }

    public String toString() {
        return "ZIPWITH";
    }
}
