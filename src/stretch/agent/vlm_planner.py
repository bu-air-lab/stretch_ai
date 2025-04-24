# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

import os
from typing import Optional

from stretch.agent.robot_agent import RobotAgent
from stretch.llms.multi_crop_openai_client import MultiCropOpenAIClient

import json

# Load the category mapping at the top of your file or in __init__
with open("/home/xin3/Desktop/stretch_ai_xin/src/stretch/config/example_cat_map.json", "r") as f:
    category_data = json.load(f)
obj_id_to_category = {v: k for k, v in category_data["obj_category_to_obj_category_id"].items()}
id_to_names = category_data["id_to_names"]
# Ensure the path to the JSON file is correct


class VLMPlanner:
    def __init__(self, agent: RobotAgent, api_key: Optional[str] = None) -> None:
        """This is a connection to a VLM for getting a plan based on language commands.

        Args:
            agent (RobotAgent): the agent
            api_key (str): the API key for the VLM. Optional; if not provided, will be read from the environment variable OPENAI_API_KEY. If not found there, will prompt the user for it.
        """

        self.agent = agent

        # Load parameters file from the agent
        self.parameters = agent.parameters
        self.voxel_map = agent.get_voxel_map()

        # TODO: put these into config
        img_size = 256
        temperature = 0.2
        max_tokens = 50
        with open(
            "/home/xin3/Desktop/stretch_ai_xin/src/stretch/llms/prompts/obj_centric_vlm.txt",
            "r",
        ) as f:
            prompt = f.read()

        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = input("You are using GPT4v for planning, please type in your openai key: ")
        self.api_key = api_key

        self.gpt_agent = MultiCropOpenAIClient(
            cfg=dict(
                img_size=img_size,
                prompt=prompt,
                api_key=self.api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        self.gpt_agent.reset()

    def set_agent(self, agent: RobotAgent) -> None:
        """Set the agent for the VLM planner.

        Args:
            agent (RobotAgent): the agent
        """
        self.agent = agent

    def plan(
        self,
        current_pose=None,
        show_prompts=False,
        show_plan=False,
        plan_file: str = "vlm_plan.txt",
        query=None,
        plan_with_reachable_instances=False,
        plan_with_scene_graph=False,
    ) -> str:
        """This is a connection to a VLM for getting a plan based on language commands.

        Args:
            current_pose(np.ndarray): the current pose of the robot
            show_prompts(bool): whether to show prompts
            show_plan(bool): whether to show the plan
            plan_file(str): the name of the file to save the plan to
            query(str): the query to send to the VLM

        Returns:
            str: the plan
        """
        world_representation = self.agent.get_object_centric_observations(
            task=query,
            current_pose=current_pose,
            show_prompts=show_prompts,
            plan_with_reachable_instances=plan_with_reachable_instances,
            plan_with_scene_graph=plan_with_scene_graph,
        )
        output = self.get_output_from_gpt(world_representation, task=query)
        if show_plan:
            import re

            import matplotlib.pyplot as plt

            if output == "explore":
                print(">>>>>> Planner cannot find a plan, the robot should explore more >>>>>>>>>")
            elif output == "gpt API error":
                print(">>>>>> there is something wrong with the planner api >>>>>>>>>")
            else:
                actions = output.split("; ")
                plt.clf()
                for action_id, action in enumerate(actions):
                    crop_id = int(re.search(r"img_(\d+)", action).group(1))
                    global_id = world_representation.object_images[crop_id].instance_id
                    instance = self.voxel_map.get_instances()[global_id]
                    category_id = getattr(instance, "category_id", None)
                    # Ensure category_id is int for lookup
                    try:
                        category_id_int = int(category_id)
                    except Exception:
                        category_id_int = category_id
                    # Get the category name from the mapping
                    category_names = id_to_names.get(str(category_id), ["unknown"])
                    category_name_str = ", ".join(category_names)

                    print(f"Action: {action}, Instance {global_id}, Category: {category_name_str}, Semantic ID: {category_id_int}")
                    plt.subplot(1, len(actions), action_id + 1)
                    plt.imshow(instance.get_best_view().get_image())
                    plt.title(f"{action.split('(')[0]}\n{category_name_str} ({global_id})\nID: {category_id_int}")
                    plt.axis("off")
                plt.suptitle(f"Task: {query}")
                plt.show()
                plt.savefig("plan.png")

        if self.parameters.get("save_vlm_plan", True):
            with open(plan_file, "w") as f:
                f.write(output)
            print(f"Task plan generated from VLMs has been written to {plan_file}")
        return actions, world_representation

    def get_output_from_gpt(self, world_rep, task: str):

        plan = self.gpt_agent.act_on_observations(world_rep, goal=task, debug_path=None)
        return plan
