package org.genesys.utils;

import com.google.gson.*;
import com.google.gson.internal.LinkedTreeMap;
import org.genesys.models.Example;
import org.genesys.models.Problem;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by yufeng on 9/12/17.
 */
public class DeepCodeDeserializer implements JsonDeserializer<Problem> {
    @Override
    public Problem deserialize(JsonElement jsonElement, Type type, JsonDeserializationContext jsonDeserializationContext) throws JsonParseException {
        Problem problem = new Problem();
        List<Example> examples = new ArrayList<>();
        final JsonObject jsonObject = jsonElement.getAsJsonObject();
        final JsonElement probName = jsonObject.get("name");
        problem.setName(probName.getAsString());
        final JsonElement probExamples = jsonObject.get("examples");

        for (JsonElement expJson : probExamples.getAsJsonArray()) {
            Example example = new Example();
            JsonElement out = expJson.getAsJsonObject().get("output");
            if (out.isJsonPrimitive())
                example.setOutput(out.getAsInt());
            else {
                List outList = parseJson(out);
                example.setOutput(outList);
            }

            List inputList = new ArrayList<>();
            for (JsonElement inputJson : expJson.getAsJsonObject().get("input").getAsJsonArray()) {
                if (inputJson.isJsonPrimitive())
                    inputList.add(inputJson.getAsInt());
                else {
                    List inputElem = parseJson(inputJson);
                    inputList.add(inputElem);
                }
            }
            example.setInput(inputList);
            examples.add(example);
        }


        problem.setExamples(examples);
        return problem;
    }

    private List parseJson(JsonElement json) {
        List list = new ArrayList();
        for (JsonElement o : json.getAsJsonArray()) {
            Number num = o.getAsNumber();
            list.add(num.intValue());
        }
        return list;
    }

}
