package org.genesys.interpreter;

import org.genesys.interpreter.morpheus.*;
import org.genesys.language.Production;
import org.genesys.models.Pair;
import org.genesys.type.AbstractType;
import org.genesys.type.Maybe;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 9/10/17.
 * The validator will do both type checking and partial evaluation.
 */
public class MorpheusValidator2 extends BaseValidatorDriver2 {

    public MorpheusValidator2(List<Production<AbstractType>> inits) {

        validators.put("root", (objects, input, ast) -> {
            Object obj = objects.get(0);
            assert obj instanceof Pair : objects;
            return (Pair) obj;
        });

        validators.put("input0", (objects, input, ast) -> new Pair<>(((List) input).get(0), new ArrayList<>()));
        validators.put("input1", (objects, input, ast) -> new Pair<>(((List) input).get(1), new ArrayList<>()));

        validators.put("spread", (objects, input, ast) -> new Spread().verify2(objects, ast));

        validators.put("select", (objects, input, ast) -> new Select().verify2(objects, ast));

        validators.put("inner_join", (objects, input, ast) -> new InnerJoin().verify2(objects, ast));

        validators.put("group_by", (objects, input, ast) -> new GroupBy().verify2(objects, ast));

        validators.put("filter", (objects, input, ast) -> new org.genesys.interpreter.morpheus.Filter().verify2(objects, ast));

        validators.put("mutate", (objects, input, ast) -> new Mutate().verify2(objects, ast));

        validators.put("summarise", (objects, input, ast) -> new Summarise().verify2(objects, ast));

        validators.put("separate", (objects, input, ast) -> new Separate().verify2(objects, ast));

        validators.put("unite", (objects, input, ast) -> new Unite().verify2(objects, ast));

        validators.put("gather", (objects, input, ast) -> new Gather().verify2(objects, ast));

        validators.put("l(a,b).(/ a b)", (objects, input, ast) -> new Pair<>(new PrimitiveBinop("/"), new ArrayList<>()));
        validators.put("l(a,b).(> a b)", (objects, input, ast) -> new Pair<>(new PrimitiveBinop(">"), new ArrayList<>()));
        validators.put("l(a,b).(< a b)", (objects, input, ast) -> new Pair<>(new PrimitiveBinop("<"), new ArrayList<>()));
        validators.put("l(a,b).(== a b)", (objects, input, ast) -> new Pair<>(new PrimitiveBinop("=="), new ArrayList<>()));
        validators.put("l(a,b).(!= a b)", (objects, input, ast) -> new Pair<>(new PrimitiveBinop("!="), new ArrayList<>()));
        ;
        validators.put("sum", (objects, input, ast) -> new Pair<>("sum", new ArrayList<>()));
        validators.put("mean", (objects, input, ast) -> new Pair<>("mean", new ArrayList<>()));
        validators.put("min", (objects, input, ast) -> new Pair<>("min", new ArrayList<>()));
        validators.put("count", (objects, input, ast) -> new Pair<>("count", new ArrayList<>()));

        for (Production<AbstractType> prod : inits) {
            validators.put(prod.function, (objects, input, ast) -> new Pair<>(prod.getValue(), new ArrayList<>()));
        }
    }

}
