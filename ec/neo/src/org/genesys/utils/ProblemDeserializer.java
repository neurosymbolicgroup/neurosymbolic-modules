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
public class ProblemDeserializer implements JsonDeserializer<Problem> {
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
            LinkedTreeMap output = new LinkedTreeMap();

            JsonObject out = expJson.getAsJsonObject().get("output").getAsJsonObject();
            List outHeader = parseJson(out.get("header"));
            output.put("header", outHeader);
            List outContent = parseJson(out.get("content"));
            output.put("content", outContent);
            example.setOutput(output);

            List<LinkedTreeMap> inputList = new ArrayList<>();
            for (JsonElement inputJson : expJson.getAsJsonObject().get("input").getAsJsonArray()) {
                LinkedTreeMap inputElem = new LinkedTreeMap();

                List inHeader = parseJson(inputJson.getAsJsonObject().get("header"));
                List inContent = parseJson(inputJson.getAsJsonObject().get("content"));
                inputElem.put("header", inHeader);
                inputElem.put("content", inContent);
                inputList.add(inputElem);
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
            if (o.toString().contains("\"")) {
                list.add(o.getAsString());
            } else {//numeric
                Number num = o.getAsNumber();
                // here you can handle double int/long values
                // and return any type you want
                // this solution will transform 3.0 float to long values
                if (num.doubleValue() % 1 == 0)
                    list.add(num.intValue());
                else {
                    list.add(num.doubleValue());
                }
            }
        }
        return list;
    }

}
