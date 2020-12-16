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
public class MorpheusValidator extends BaseValidatorDriver {

    public MorpheusValidator(List<Production<AbstractType>> inits) {

        validators.put("root", (objects, input) -> {
            Object obj = objects.get(0);
            assert obj instanceof Pair : objects;
//            if (obj instanceof Unop)
//                return new Maybe<>(((Unop) objects.get(0)).apply(input));
//            else
//                return new Maybe<>(obj);
            return (Pair) obj;
        });

        validators.put("input0", (objects, input) -> new Pair<>(true, new Maybe<>(((List) input).get(0))));
        validators.put("input1", (objects, input) -> new Pair<>(true, new Maybe<>(((List) input).get(1))));

        validators.put("spread", (objects, input) -> new Spread().verify(objects));

        validators.put("select", (objects, input) -> new Select().verify(objects));

        validators.put("inner_join", (objects, input) -> new InnerJoin().verify(objects));

        validators.put("group_by", (objects, input) -> new GroupBy().verify(objects));

        validators.put("filter", (objects, input) -> new org.genesys.interpreter.morpheus.Filter().verify(objects));

        validators.put("mutate", (objects, input) -> new Mutate().verify(objects));

        validators.put("summarise", (objects, input) -> new Summarise().verify(objects));

        validators.put("separate", (objects, input) -> new Separate().verify(objects));

        validators.put("unite", (objects, input) -> new Unite().verify(objects));

        validators.put("gather", (objects, input) -> new Gather().verify(objects));

        validators.put("l(a,b).(/ a b)", (objects, input) -> new Pair<>(true, new Maybe<>(new PrimitiveBinop("/"))));
        validators.put("l(a,b).(> a b)", (objects, input) -> new Pair<>(true, new Maybe<>(new PrimitiveBinop(">"))));
        validators.put("l(a,b).(< a b)", (objects, input) -> new Pair<>(true, new Maybe<>(new PrimitiveBinop("<"))));
        validators.put("l(a,b).(== a b)", (objects, input) -> new Pair<>(true, new Maybe<>(new PrimitiveBinop("=="))));
        validators.put("l(a,b).(!= a b)", (objects, input) -> new Pair<>(true, new Maybe<>(new PrimitiveBinop("!="))));
        validators.put("l(a).(> a b)", (objects, input) -> new Pair<>(true, new Maybe<>(new PrimitiveUnop(">", objects.get(0)))));
        validators.put("l(a).(< a b)", (objects, input) -> new Pair<>(true, new Maybe<>(new PrimitiveUnop("<", objects.get(0)))));
        validators.put("l(a).(== a b)", (objects, input) -> new Pair<>(true, new Maybe<>(new PrimitiveUnop("==", objects.get(0)))));

        validators.put("sum", (objects, input) -> new Pair<>(true, new Maybe<>("sum")));
        validators.put("mean", (objects, input) -> new Pair<>(true, new Maybe<>("mean")));
        validators.put("min", (objects, input) -> new Pair<>(true, new Maybe<>("min")));
        validators.put("count", (objects, input) -> new Pair<>(true, new Maybe<>("count")));

        for (Production<AbstractType> prod : inits) {
            validators.put(prod.function, (objects, input) -> new Pair<>(true, new Maybe<>(prod.getValue())));
        }
    }

}
